#!/usr/bin/env python3
"""Offline paper diagnostic for boundary error, correction activity, and topology.

This tool reads existing correction artifacts. It never runs SDN, Original GDC,
or Range-GDC correction. RGC graph construction is reconstructed with the
production builder, but no graph solve is performed.

RGC edge weights and Original-GDC reconstruction coefficients have different
semantics. Reported graph quantities are method-specific normalized influence
magnitudes, not directly equivalent edge-weight probabilities. Exact Original
GDC graph mapping is intentionally reported as unsupported unless an exact
camera-node to canonical-range-cell correspondence is available; this tool does
not invent an approximate correspondence.

Correction ``harm_rate`` is harmed/all evaluated pixels; ``changed_harm_rate``
is changed-and-harmed/changed pixels. Activity and GT-error outcome are separate
classifications.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.evaluate_range_metrics import error_metrics  # noqa: E402
from range_gdc.range_gdc import (  # noqa: E402
    build_spherical_graph_laplacian,
    valid_range_mask,
)
from range_gdc.range_main_batch import npy_map, read_split_scene_ids  # noqa: E402


REGIONS = ("boundary", "interior")
DISTANCE_BINS = ("0", "1", "2", "3", "4+")
HARM_TOL = 1e-6
DEFAULT_CORRECTION_TOL = 1e-3
GDC_UNSUPPORTED_STATUS = "unsupported_insufficient_exact_mapping"


def gt_boundary_mask(gt_range, range_min=0.1, range_max=80.0,
                     boundary_log_thr=0.2):
    """Mark endpoints of GT-valid 4-neighbor log-range discontinuities.

    Horizontal adjacency is periodic; vertical adjacency is not. The default
    threshold is diagnostic, not a tuned paper constant.
    """
    gt = np.asarray(gt_range, dtype=np.float64)
    if gt.ndim != 2:
        raise ValueError("GT range must be a 2D array")
    if boundary_log_thr < 0:
        raise ValueError("boundary_log_thr must be nonnegative")
    valid = valid_range_mask(gt, range_min, range_max)
    log_gt = np.zeros_like(gt, dtype=np.float64)
    log_gt[valid] = np.log(gt[valid])
    boundary = np.zeros_like(valid)
    height, width = gt.shape

    if width > 1:
        right_valid = valid & np.roll(valid, -1, axis=1)
        right_jump = np.abs(log_gt - np.roll(log_gt, -1, axis=1))
        pair = right_valid & (right_jump > float(boundary_log_thr))
        boundary |= pair
        boundary |= np.roll(pair, 1, axis=1)

    if height > 1:
        pair = (
            valid[:-1]
            & valid[1:]
            & (np.abs(log_gt[:-1] - log_gt[1:]) > float(boundary_log_thr))
        )
        boundary[:-1] |= pair
        boundary[1:] |= pair
    return boundary


def periodic_boundary_distance(boundary_mask):
    """Euclidean grid distance to a boundary with horizontal periodicity."""
    boundary = np.asarray(boundary_mask, dtype=bool)
    if boundary.ndim != 2:
        raise ValueError("boundary_mask must be 2D")
    if not np.any(boundary):
        return np.full(boundary.shape, np.inf, dtype=np.float64)
    tiled = np.tile(boundary, (1, 3))
    tiled_distance = distance_transform_edt(~tiled)
    width = boundary.shape[1]
    return tiled_distance[:, width:2 * width]


def hidden_rows_mask(shape, source_rows):
    hidden = np.ones(shape, dtype=bool)
    rows = np.asarray(source_rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("source_rows must be a non-empty 1D list")
    if np.any(rows < 0) or np.any(rows >= shape[0]):
        raise ValueError(f"source_rows are outside [0, {shape[0]})")
    hidden[rows, :] = False
    return hidden


def common_hidden_valid_mask(gt, predictions, source_rows,
                             range_min=0.1, range_max=80.0):
    """Logical equivalent of evaluator common_hidden_valid for these methods."""
    gt = np.asarray(gt)
    common = valid_range_mask(gt, range_min, range_max)
    common &= hidden_rows_mask(gt.shape, source_rows)
    for prediction in predictions:
        prediction = np.asarray(prediction)
        if prediction.shape != gt.shape:
            raise ValueError("All predictions must share the GT shape")
        common &= valid_range_mask(prediction, range_min, range_max)
    return common


def region_mask(common_mask, boundary_distance, region, boundary_radius=1):
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be nonnegative")
    if region == "boundary":
        return common_mask & (boundary_distance <= float(boundary_radius))
    if region == "interior":
        return common_mask & (boundary_distance >= 4.0)
    raise ValueError(f"Unknown region: {region}")


def boundary_distance_bin_mask(common_mask, boundary_distance, label):
    if label == "4+":
        return common_mask & (boundary_distance >= 4.0)
    distance = int(label)
    return common_mask & (boundary_distance >= distance) & (boundary_distance < distance + 1)


def comparison_metrics(prediction, gt, raw, mask, harm_tol=HARM_TOL):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    errors = prediction[mask] - gt[mask]
    raw_abs = np.abs(raw[mask] - gt[mask])
    method_abs = np.abs(errors)
    metrics = error_metrics(errors)
    if errors.size == 0:
        harm_rate = math.nan
        mean_harm = math.nan
        mean_improvement = math.nan
        delta = np.array([], dtype=np.float64)
    else:
        delta = method_abs - raw_abs
        harmed = delta > float(harm_tol)
        improved = delta < -float(harm_tol)
        harm_rate = float(np.mean(harmed))
        mean_harm = float(np.mean(delta[harmed])) if np.any(harmed) else 0.0
        mean_improvement = float(np.mean(-delta[improved])) if np.any(improved) else 0.0
    return {
        "count": int(errors.size),
        **metrics,
        "harm_rate": harm_rate,
        "mean_harm_magnitude": mean_harm,
        "mean_improvement_magnitude": mean_improvement,
    }, {"errors": errors, "delta": delta}


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else math.nan


def _safe_median(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if values.size else math.nan


def _correction_activity_from_vectors(signed_correction, error_delta,
                                      correction_tol=DEFAULT_CORRECTION_TOL,
                                      error_tol=HARM_TOL):
    """Summarize correction activity and GT-error outcomes on one domain.

    Activity is independent of outcome: ``changed`` uses correction magnitude,
    while improved/harmed/neutral use the change in absolute GT error.
    """
    if not np.isfinite(correction_tol) or correction_tol < 0:
        raise ValueError("correction_tol must be finite and nonnegative")
    if not np.isfinite(error_tol) or error_tol < 0:
        raise ValueError("error_tol must be finite and nonnegative")
    signed = np.asarray(signed_correction, dtype=np.float64).reshape(-1)
    delta = np.asarray(error_delta, dtype=np.float64).reshape(-1)
    if signed.shape != delta.shape:
        raise ValueError("signed_correction and error_delta must have the same shape")
    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(delta)):
        raise ValueError("activity vectors must be finite")

    absolute = np.abs(signed)
    changed = absolute > float(correction_tol)
    improved = delta < -float(error_tol)
    harmed = delta > float(error_tol)
    neutral = ~(improved | harmed)
    unchanged = ~changed
    changed_improved = changed & improved
    changed_harmed = changed & harmed
    changed_neutral = changed & neutral

    pixels = int(delta.size)
    changed_pixels = int(changed.sum())
    unchanged_pixels = int(unchanged.sum())
    improved_pixels = int(improved.sum())
    harmed_pixels = int(harmed.sum())
    neutral_pixels = int(neutral.sum())
    changed_improved_pixels = int(changed_improved.sum())
    changed_harmed_pixels = int(changed_harmed.sum())
    changed_neutral_pixels = int(changed_neutral.sum())
    if changed_pixels + unchanged_pixels != pixels:
        raise RuntimeError("changed/unchanged activity identity failed")
    if improved_pixels + harmed_pixels + neutral_pixels != pixels:
        raise RuntimeError("improved/harmed/neutral activity identity failed")
    if (
        changed_improved_pixels
        + changed_harmed_pixels
        + changed_neutral_pixels
        != changed_pixels
    ):
        raise RuntimeError("changed outcome activity identity failed")

    positive_improvement = np.maximum(-delta, 0.0)
    positive_harm = np.maximum(delta, 0.0)
    total_improvement = float(np.sum(positive_improvement))
    total_harm = float(np.sum(positive_harm))
    net_error_reduction = total_improvement - total_harm
    if total_harm > 0.0:
        improvement_to_harm_ratio = total_improvement / total_harm
    elif total_improvement > 0.0:
        improvement_to_harm_ratio = math.inf
    else:
        improvement_to_harm_ratio = math.nan
    changed_absolute = absolute[changed]
    return {
        "pixels": pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "changed_ratio": _safe_ratio(changed_pixels, pixels),
        "unchanged_ratio": _safe_ratio(unchanged_pixels, pixels),
        "improved_pixels": improved_pixels,
        "harmed_pixels": harmed_pixels,
        "neutral_pixels": neutral_pixels,
        "improved_ratio": _safe_ratio(improved_pixels, pixels),
        "harm_rate": _safe_ratio(harmed_pixels, pixels),
        "neutral_ratio": _safe_ratio(neutral_pixels, pixels),
        "changed_improved_pixels": changed_improved_pixels,
        "changed_harmed_pixels": changed_harmed_pixels,
        "changed_neutral_pixels": changed_neutral_pixels,
        "changed_improvement_rate": _safe_ratio(
            changed_improved_pixels, changed_pixels
        ),
        "changed_harm_rate": _safe_ratio(changed_harmed_pixels, changed_pixels),
        "changed_neutral_rate": _safe_ratio(
            changed_neutral_pixels, changed_pixels
        ),
        "mean_abs_correction": _safe_mean(absolute),
        "median_abs_correction": _safe_median(absolute),
        "p90_abs_correction": (
            float(np.percentile(absolute, 90)) if pixels else math.nan
        ),
        "p95_abs_correction": (
            float(np.percentile(absolute, 95)) if pixels else math.nan
        ),
        "mean_abs_correction_changed_only": _safe_mean(changed_absolute),
        "median_abs_correction_changed_only": _safe_median(changed_absolute),
        "total_improvement": total_improvement,
        "total_harm": total_harm,
        "net_error_reduction": net_error_reduction,
        "mean_net_error_reduction": _safe_ratio(net_error_reduction, pixels),
        "improvement_to_harm_ratio": improvement_to_harm_ratio,
        "mean_signed_correction": _safe_mean(signed),
        "median_signed_correction": _safe_median(signed),
    }


def correction_activity_metrics(prediction, gt, raw, mask,
                                correction_tol=DEFAULT_CORRECTION_TOL,
                                error_tol=HARM_TOL):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != gt.shape or raw.shape != gt.shape or mask.shape != gt.shape:
        raise ValueError("prediction, GT, raw, and mask must share a shape")
    signed = prediction[mask] - raw[mask]
    error_delta = (
        np.abs(prediction[mask] - gt[mask])
        - np.abs(raw[mask] - gt[mask])
    )
    metrics = _correction_activity_from_vectors(
        signed, error_delta, correction_tol, error_tol
    )
    return metrics, {"signed_correction": signed, "error_delta": error_delta}


def rgc_graph_quality_rows(raw_range, gt_range, projection, args):
    """Build production RGC edges once and score full/angular-only influence."""
    raw = np.asarray(raw_range, dtype=np.float64)
    gt = np.asarray(gt_range, dtype=np.float64)
    guide_valid = valid_range_mask(raw, args.range_min, args.range_max)
    node_rows, node_cols = np.where(guide_valid)
    node_id = np.full(raw.shape, -1, dtype=np.int32)
    node_id[node_rows, node_cols] = np.arange(node_rows.size, dtype=np.int32)
    _, _, debug = build_spherical_graph_laplacian(
        raw,
        guide_valid,
        node_id,
        node_rows,
        node_cols,
        vertical_centers_deg=projection["vertical_centers_deg"],
        azimuth_centers_deg=projection["azimuth_centers_deg"],
        azimuth_mode=projection["azimuth_mode"],
        neighbor=args.neighbor,
        edge_spatial_mode=args.edge_spatial_mode,
        sigma_angular=args.sigma_angular,
        sigma_tangent=args.sigma_tangent,
        sigma_log_range=args.sigma_log_range,
        max_log_range_diff=args.max_log_range_diff,
    )
    edge_i = debug["edge_i"]
    edge_j = debug["edge_j"]
    edge_rows_i, edge_cols_i = node_rows[edge_i], node_cols[edge_i]
    edge_rows_j, edge_cols_j = node_rows[edge_j], node_cols[edge_j]
    gt_valid = valid_range_mask(gt, args.range_min, args.range_max)
    mapped = gt_valid[edge_rows_i, edge_cols_i] & gt_valid[edge_rows_j, edge_cols_j]
    gt_log_diff = np.full(edge_i.shape, np.nan, dtype=np.float64)
    if np.any(mapped):
        gt_log_diff[mapped] = np.abs(
            np.log(gt[edge_rows_i[mapped], edge_cols_i[mapped]])
            - np.log(gt[edge_rows_j[mapped], edge_cols_j[mapped]])
        )
    cross = mapped & (gt_log_diff > float(args.boundary_log_thr))
    same = mapped & ~cross

    rows = []
    weight_specs = (
        ("rgc_angular_only_weight", debug["spatial_weight"]),
        ("rgc_full", debug["edge_weight"]),
    )
    for method, influence in weight_specs:
        valid_influence = influence[mapped]
        cross_influence = influence[cross]
        row = {
            "method": method,
            "status": "ok",
            "N_edges_total": int(edge_i.size),
            "gt_valid_edges": int(mapped.sum()),
            "cross_boundary_edges": int(cross.sum()),
            "cross_boundary_edge_ratio": _safe_ratio(int(cross.sum()), int(mapped.sum())),
            "normalized_cross_boundary_influence": _safe_ratio(
                float(np.sum(cross_influence)), float(np.sum(valid_influence))
            ),
            "mean_same_surface_influence": _safe_mean(influence[same]),
            "mean_cross_boundary_influence": _safe_mean(cross_influence),
            "median_same_surface_influence": _safe_median(influence[same]),
            "median_cross_boundary_influence": _safe_median(cross_influence),
            "mean_range_gate_same_surface": _safe_mean(debug["range_gate"][same]),
            "mean_range_gate_cross_boundary": _safe_mean(debug["range_gate"][cross]),
            "mapping_ratio": _safe_ratio(int(mapped.sum()), int(edge_i.size)),
            "edge_endpoint_digest": _edge_digest(edge_i, edge_j),
        }
        rows.append(row)
    if rows[0]["N_edges_total"] != rows[1]["N_edges_total"]:
        raise RuntimeError("Angular-only and full RGC edge counts differ")
    if rows[0]["edge_endpoint_digest"] != rows[1]["edge_endpoint_digest"]:
        raise RuntimeError("Angular-only and full RGC edge endpoints differ")
    return rows, debug


def _edge_digest(edge_i, edge_j):
    import hashlib
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(edge_i).tobytes())
    digest.update(np.ascontiguousarray(edge_j).tobytes())
    return digest.hexdigest()


def unsupported_gdc_graph_row():
    return {
        "method": "gdc",
        "status": GDC_UNSUPPORTED_STATUS,
        "N_edges_total": 0,
        "gt_valid_edges": 0,
        "cross_boundary_edges": 0,
        "cross_boundary_edge_ratio": math.nan,
        "normalized_cross_boundary_influence": math.nan,
        "mean_same_surface_influence": math.nan,
        "mean_cross_boundary_influence": math.nan,
        "median_same_surface_influence": math.nan,
        "median_cross_boundary_influence": math.nan,
        "mean_range_gate_same_surface": math.nan,
        "mean_range_gate_cross_boundary": math.nan,
        "mapping_ratio": 0.0,
        "edge_endpoint_digest": "",
    }


def aggregate_boundary_rows(accumulators, frame_count):
    output = []
    for (method, region), item in sorted(accumulators.items()):
        errors = np.concatenate(item["errors"]) if item["errors"] else np.array([])
        delta = np.concatenate(item["delta"]) if item["delta"] else np.array([])
        metrics = error_metrics(errors)
        harmed = delta > HARM_TOL
        improved = delta < -HARM_TOL
        output.append({
            "method": method,
            "region": region,
            "frames": frame_count,
            "pixels": int(errors.size),
            **metrics,
            "harm_rate": float(np.mean(harmed)) if delta.size else math.nan,
            "mean_harm_magnitude": float(np.mean(delta[harmed])) if np.any(harmed) else (0.0 if delta.size else math.nan),
            "mean_improvement_magnitude": float(np.mean(-delta[improved])) if np.any(improved) else (0.0 if delta.size else math.nan),
        })
    return output


def aggregate_distance_rows(accumulators, frame_count):
    output = []
    order = {label: index for index, label in enumerate(DISTANCE_BINS)}
    for (method, label), item in sorted(
        accumulators.items(), key=lambda value: (value[0][0], order[value[0][1]])
    ):
        errors = np.concatenate(item) if item else np.array([])
        metrics = error_metrics(errors)
        output.append({
            "method": method,
            "boundary_distance_bin": label,
            "frames": frame_count,
            "pixels": int(errors.size),
            "mae": metrics["mae"],
            "median_abs": metrics["median_abs"],
            "p90_abs": metrics["p90_abs"],
        })
    return output


def aggregate_activity_rows(accumulators, frame_count, correction_tol,
                            distance_profile=False):
    output = []
    distance_order = {label: index for index, label in enumerate(DISTANCE_BINS)}
    items = accumulators.items()
    if distance_profile:
        items = sorted(
            items, key=lambda value: (value[0][0], distance_order[value[0][1]])
        )
    else:
        items = sorted(items)
    for (method, region), item in items:
        signed = (
            np.concatenate(item["signed_correction"])
            if item["signed_correction"] else np.array([], dtype=np.float64)
        )
        error_delta = (
            np.concatenate(item["error_delta"])
            if item["error_delta"] else np.array([], dtype=np.float64)
        )
        metrics = _correction_activity_from_vectors(
            signed, error_delta, correction_tol, HARM_TOL
        )
        if distance_profile:
            row = {
                "method": method,
                "boundary_distance_bin": region,
                "frames": frame_count,
                **metrics,
            }
        else:
            row = {
                "method": method,
                "region": region,
                "frames": frame_count,
                **metrics,
            }
        output.append(row)
    return output


def aggregate_graph_rows(rows, frame_count):
    output = []
    for method in ("gdc", "rgc_angular_only_weight", "rgc_full"):
        method_rows = [row for row in rows if row["method"] == method]
        if method == "gdc":
            result = unsupported_gdc_graph_row()
            result.update({"frames": frame_count})
            output.append(result)
            continue
        total = sum(int(row["N_edges_total"]) for row in method_rows)
        mapped = sum(int(row["gt_valid_edges"]) for row in method_rows)
        cross = sum(int(row["cross_boundary_edges"]) for row in method_rows)
        influence_total = sum(float(row["_influence_total"]) for row in method_rows)
        influence_cross = sum(float(row["_influence_cross"]) for row in method_rows)
        same_values = np.concatenate([row["_same_values"] for row in method_rows if row["_same_values"].size]) if any(row["_same_values"].size for row in method_rows) else np.array([])
        cross_values = np.concatenate([row["_cross_values"] for row in method_rows if row["_cross_values"].size]) if any(row["_cross_values"].size for row in method_rows) else np.array([])
        same_gate = np.concatenate([row["_same_gate"] for row in method_rows if row["_same_gate"].size]) if any(row["_same_gate"].size for row in method_rows) else np.array([])
        cross_gate = np.concatenate([row["_cross_gate"] for row in method_rows if row["_cross_gate"].size]) if any(row["_cross_gate"].size for row in method_rows) else np.array([])
        output.append({
            "method": method,
            "status": "ok",
            "frames": frame_count,
            "N_edges_total": total,
            "gt_valid_edges": mapped,
            "cross_boundary_edges": cross,
            "cross_boundary_edge_ratio": _safe_ratio(cross, mapped),
            "normalized_cross_boundary_influence": _safe_ratio(influence_cross, influence_total),
            "mean_same_surface_influence": _safe_mean(same_values),
            "mean_cross_boundary_influence": _safe_mean(cross_values),
            "median_same_surface_influence": _safe_median(same_values),
            "median_cross_boundary_influence": _safe_median(cross_values),
            "mean_range_gate_same_surface": _safe_mean(same_gate),
            "mean_range_gate_cross_boundary": _safe_mean(cross_gate),
            "mapping_ratio": _safe_ratio(mapped, total),
            "edge_endpoint_digest": "per_frame_verified_equal",
        })
    return output


def _attach_graph_accumulators(row, debug, gt, node_rows, node_cols, threshold,
                               range_min=0.1, range_max=80.0):
    edge_i, edge_j = debug["edge_i"], debug["edge_j"]
    ri, ci = node_rows[edge_i], node_cols[edge_i]
    rj, cj = node_rows[edge_j], node_cols[edge_j]
    gt_valid = valid_range_mask(gt, range_min, range_max)
    mapped = gt_valid[ri, ci] & gt_valid[rj, cj]
    cross = np.zeros(edge_i.shape, dtype=bool)
    cross[mapped] = np.abs(np.log(gt[ri[mapped], ci[mapped]]) - np.log(gt[rj[mapped], cj[mapped]])) > threshold
    same = mapped & ~cross
    influence = debug["spatial_weight"] if row["method"] == "rgc_angular_only_weight" else debug["edge_weight"]
    row["_influence_total"] = float(np.sum(influence[mapped]))
    row["_influence_cross"] = float(np.sum(influence[cross]))
    row["_same_values"] = influence[same]
    row["_cross_values"] = influence[cross]
    row["_same_gate"] = debug["range_gate"][same]
    row["_cross_gate"] = debug["range_gate"][cross]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if not key.startswith("_") and key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_projection_meta(args):
    if args.projection_meta_path:
        path = Path(args.projection_meta_path)
    else:
        candidates = sorted(Path(args.meta_dir).glob("**/projection_meta.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"--meta-dir must contain exactly one projection_meta.npz; found {len(candidates)}"
            )
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as meta:
        required = ("vertical_centers_deg", "azimuth_centers_deg")
        missing = [name for name in required if name not in meta.files]
        if missing:
            raise ValueError(f"Projection metadata missing: {missing}")
        projection = {
            "path": str(path.resolve()),
            "vertical_centers_deg": meta["vertical_centers_deg"].astype(np.float64),
            "azimuth_centers_deg": meta["azimuth_centers_deg"].astype(np.float64),
            "azimuth_mode": str(meta["azimuth_mode"].item()) if "azimuth_mode" in meta.files else "full_360_front_centered",
            "selected_rows": meta["selected_rows"].astype(np.int64).tolist() if "selected_rows" in meta.files else None,
        }
        for name in (
            "vmin_deg", "vmax_deg", "azimuth_min_deg", "azimuth_max_deg",
            "vertical_resolution_deg", "horizontal_resolution_deg",
        ):
            if name in meta.files:
                projection[name] = float(meta[name].item())
    projection["shape"] = (
        len(projection["vertical_centers_deg"]), len(projection["azimuth_centers_deg"])
    )
    return projection


def _is_relative_to(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_dir(output_dir, input_dirs):
    output = Path(output_dir).resolve()
    for directory in input_dirs:
        source = Path(directory).resolve()
        if _is_relative_to(output, source):
            raise ValueError(f"output-dir must not be inside an input directory: {source}")
        if _is_relative_to(source, output):
            raise ValueError(f"input directory must not be inside output-dir: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output-dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def discover_inputs(args, scene_ids):
    directories = {
        "raw_sdn": args.raw_range_dir,
        "gt": args.gt_range_dir,
        "range_gdc": args.rgc_range_dir,
        "original_gdc": args.gdc_range_dir,
        "anchor": args.anchor_range_dir,
    }
    mappings = {}
    for name, directory in directories.items():
        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(path)
        mapping = npy_map(str(path))
        missing = [scene_id for scene_id in scene_ids if scene_id not in mapping]
        if missing:
            raise FileNotFoundError(f"{name} missing {len(missing)} frames, sample={missing[:5]}")
        mappings[name] = mapping
    return directories, mappings


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def analyze(args):
    method_specs = (
        ("raw_sdn", "raw_sdn"),
        (args.gdc_label, "original_gdc"),
        ("range_gdc", "range_gdc"),
    )
    projection = resolve_projection_meta(args)
    source_rows = args.source_rows or projection["selected_rows"] or [5, 7, 9, 11]
    source_rows = [int(value) for value in source_rows]
    scene_ids = read_split_scene_ids(args.split_file)
    if args.max_items is not None:
        scene_ids = scene_ids[:args.max_items]
    directories, mappings = discover_inputs(args, scene_ids)
    output_dir = validate_output_dir(args.output_dir, directories.values())

    boundary_rows = []
    distance_rows = []
    graph_rows = []
    boundary_acc = defaultdict(lambda: {"errors": [], "delta": []})
    distance_acc = defaultdict(list)
    activity_rows = []
    activity_acc = defaultdict(
        lambda: {"signed_correction": [], "error_delta": []}
    )
    activity_distance_acc = defaultdict(
        lambda: {"signed_correction": [], "error_delta": []}
    )
    warnings = []
    boundary_pixels = 0
    common_pixels = 0

    for scene_id in scene_ids:
        arrays = {
            name: np.load(mapping[scene_id]).astype(np.float32)
            for name, mapping in mappings.items()
        }
        for name, array in arrays.items():
            if array.shape != projection["shape"]:
                raise ValueError(f"{scene_id} {name}: {array.shape} != {projection['shape']}")
        anchor_outside = arrays["anchor"].copy()
        anchor_outside[np.asarray(source_rows), :] = 0.0
        if np.any(valid_range_mask(anchor_outside, args.range_min, args.range_max)):
            raise ValueError(f"{scene_id}: canonical anchor contains values outside source rows")

        predictions = [arrays[array_name] for _, array_name in method_specs]
        common = common_hidden_valid_mask(
            arrays["gt"], predictions, source_rows, args.range_min, args.range_max
        )
        boundary = gt_boundary_mask(
            arrays["gt"], args.range_min, args.range_max, args.boundary_log_thr
        )
        distance = periodic_boundary_distance(boundary)
        boundary_region = region_mask(common, distance, "boundary", args.boundary_radius)
        boundary_pixels += int(boundary_region.sum())
        common_pixels += int(common.sum())

        for method, array_name in method_specs:
            prediction = arrays[array_name]
            for region in REGIONS:
                mask = region_mask(common, distance, region, args.boundary_radius)
                metrics, accum = comparison_metrics(
                    prediction, arrays["gt"], arrays["raw_sdn"], mask
                )
                row = {"frame_id": scene_id, "method": method, "region": region, **metrics}
                boundary_rows.append(row)
                if accum["errors"].size:
                    boundary_acc[(method, region)]["errors"].append(accum["errors"])
                    boundary_acc[(method, region)]["delta"].append(accum["delta"])
                else:
                    boundary_acc[(method, region)]
                activity_metrics, activity_vectors = correction_activity_metrics(
                    prediction,
                    arrays["gt"],
                    arrays["raw_sdn"],
                    mask,
                    args.correction_tol,
                    HARM_TOL,
                )
                activity_rows.append({
                    "frame_id": scene_id,
                    "method": method,
                    "region": region,
                    **activity_metrics,
                })
                if activity_vectors["signed_correction"].size:
                    activity_acc[(method, region)]["signed_correction"].append(
                        activity_vectors["signed_correction"]
                    )
                    activity_acc[(method, region)]["error_delta"].append(
                        activity_vectors["error_delta"]
                    )
                else:
                    activity_acc[(method, region)]
            for label in DISTANCE_BINS:
                mask = boundary_distance_bin_mask(common, distance, label)
                errors = prediction[mask].astype(np.float64) - arrays["gt"][mask]
                metrics = error_metrics(errors)
                distance_rows.append({
                    "frame_id": scene_id, "method": method,
                    "boundary_distance_bin": label, "pixels": int(errors.size),
                    "mae": metrics["mae"], "median_abs": metrics["median_abs"],
                    "p90_abs": metrics["p90_abs"],
                })
                if errors.size:
                    distance_acc[(method, label)].append(errors)
                else:
                    distance_acc[(method, label)]
                _, activity_vectors = correction_activity_metrics(
                    prediction,
                    arrays["gt"],
                    arrays["raw_sdn"],
                    mask,
                    args.correction_tol,
                    HARM_TOL,
                )
                if activity_vectors["signed_correction"].size:
                    activity_distance_acc[(method, label)][
                        "signed_correction"
                    ].append(activity_vectors["signed_correction"])
                    activity_distance_acc[(method, label)]["error_delta"].append(
                        activity_vectors["error_delta"]
                    )
                else:
                    activity_distance_acc[(method, label)]

        rgc_rows, debug = rgc_graph_quality_rows(arrays["raw_sdn"], arrays["gt"], projection, args)
        node_rows, node_cols = np.where(valid_range_mask(arrays["raw_sdn"], args.range_min, args.range_max))
        for row in rgc_rows:
            row["frame_id"] = scene_id
            _attach_graph_accumulators(
                row, debug, arrays["gt"], node_rows, node_cols,
                args.boundary_log_thr, args.range_min, args.range_max,
            )
            graph_rows.append(row)
        gdc_row = unsupported_gdc_graph_row()
        gdc_row["frame_id"] = scene_id
        graph_rows.append(gdc_row)

    boundary_summary = aggregate_boundary_rows(boundary_acc, len(scene_ids))
    distance_summary = aggregate_distance_rows(distance_acc, len(scene_ids))
    graph_summary = aggregate_graph_rows(graph_rows, len(scene_ids))
    activity_summary = aggregate_activity_rows(
        activity_acc, len(scene_ids), args.correction_tol
    )
    activity_distance_summary = aggregate_activity_rows(
        activity_distance_acc,
        len(scene_ids),
        args.correction_tol,
        distance_profile=True,
    )
    boundary_fraction = _safe_ratio(boundary_pixels, common_pixels)
    if np.isfinite(boundary_fraction) and (boundary_fraction < 0.01 or boundary_fraction > 0.50):
        warnings.append(
            f"boundary region fraction {boundary_fraction:.6f} is outside diagnostic sanity range [0.01, 0.50]"
        )
    full = next(row for row in graph_summary if row["method"] == "rgc_full")
    angular = next(row for row in graph_summary if row["method"] == "rgc_angular_only_weight")
    if full["N_edges_total"] != angular["N_edges_total"]:
        raise RuntimeError("RGC full/angular-only aggregate topology differs")
    if not (
        np.isfinite(full["mean_range_gate_cross_boundary"])
        and np.isfinite(full["mean_range_gate_same_surface"])
        and full["mean_range_gate_cross_boundary"] < full["mean_range_gate_same_surface"]
    ):
        warnings.append("mean RGC range gate is not lower on GT cross-boundary edges")
    warnings.append(
        "Original GDC graph quality unsupported: range artifacts do not provide exact camera-node to canonical GT-cell mapping"
    )

    write_csv(output_dir / "boundary_per_frame.csv", boundary_rows)
    write_csv(output_dir / "boundary_summary.csv", boundary_summary)
    write_csv(output_dir / "boundary_distance_per_frame.csv", distance_rows)
    write_csv(output_dir / "boundary_distance_summary.csv", distance_summary)
    write_csv(output_dir / "graph_edge_per_frame.csv", graph_rows)
    write_csv(output_dir / "graph_edge_summary.csv", graph_summary)
    write_csv(output_dir / "correction_activity_per_frame.csv", activity_rows)
    write_csv(output_dir / "correction_activity_summary.csv", activity_summary)
    write_csv(
        output_dir / "correction_activity_distance_summary.csv",
        activity_distance_summary,
    )

    metadata = {
        "analysis_kind": "offline_topology_diagnostic_not_a_correction_method",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "split_file": str(Path(args.split_file).resolve()),
        "number_of_frames": len(scene_ids),
        "source_rows": source_rows,
        "range_min": args.range_min,
        "range_max": args.range_max,
        "boundary_log_thr": args.boundary_log_thr,
        "boundary_radius": args.boundary_radius,
        "boundary_region_pixels": boundary_pixels,
        "common_hidden_valid_pixels": common_pixels,
        "boundary_pixel_fraction": boundary_fraction,
        "harm_tolerance": HARM_TOL,
        "correction_tolerance": args.correction_tol,
        "error_tolerance": HARM_TOL,
        "gdc_label": args.gdc_label,
        "activity_semantics": {
            "changed": "abs(method_range - raw_range) > correction_tolerance",
            "unchanged": "abs(method_range - raw_range) <= correction_tolerance",
            "outcome": "improved/harmed compare absolute GT error using error_tolerance; neutral is the complement",
            "harm_rate": "harmed pixels / all evaluated pixels",
            "changed_harm_rate": "changed-and-harmed pixels / changed pixels",
            "empty_changed_statistics": "changed-only statistics and conditional rates are NaN",
            "improvement_to_harm_ratio": "inf when only improvement is positive; NaN when both totals are zero",
        },
        "rgc_graph_parameters": {
            "neighbor": args.neighbor,
            "edge_spatial_mode": args.edge_spatial_mode,
            "sigma_angular": args.sigma_angular,
            "sigma_tangent": args.sigma_tangent,
            "sigma_log_range": args.sigma_log_range,
            "max_log_range_diff": args.max_log_range_diff,
        },
        "input_directories": {name: str(Path(path).resolve()) for name, path in directories.items()},
        "projection_meta_path": projection["path"],
        "rgc_angular_only_note": "same full-RGC edge endpoints; spatial_weight replaces spatial_weight*range_gate; not a correction ablation",
        "influence_semantics_warning": "RGC edge_weight and GDC abs(W_ij) are method-specific normalized influence magnitudes, not equivalent probabilities",
        "gdc_graph_status": GDC_UNSUPPORTED_STATUS,
        "warnings": warnings,
    }
    with (output_dir / "analysis_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    print(f"Analyzed {len(scene_ids)} frame(s)")
    print(f"boundary_pixel_fraction={boundary_fraction}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Saved: {output_dir}")
    return {
        "boundary_summary": boundary_summary,
        "distance_summary": distance_summary,
        "graph_summary": graph_summary,
        "activity_summary": activity_summary,
        "activity_distance_summary": activity_distance_summary,
        "metadata": metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-range-dir", required=True)
    parser.add_argument("--gt-range-dir", required=True)
    parser.add_argument("--rgc-range-dir", required=True)
    parser.add_argument("--gdc-range-dir", required=True)
    parser.add_argument("--gdc-label", default="original_gdc")
    parser.add_argument("--anchor-range-dir", required=True)
    meta = parser.add_mutually_exclusive_group(required=True)
    meta.add_argument("--projection-meta-path")
    meta.add_argument("--meta-dir")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rows", type=int, nargs="+", default=None)
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--boundary-log-thr", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument(
        "--correction-tol", type=float, default=DEFAULT_CORRECTION_TOL
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--neighbor", choices=["angular_grid4", "angular_grid8"], default="angular_grid8")
    parser.add_argument("--edge-spatial-mode", choices=["angular", "tangent"], default="angular")
    parser.add_argument("--sigma-angular", type=float, default=0.01)
    parser.add_argument("--sigma-tangent", type=float, default=1.0)
    parser.add_argument("--sigma-log-range", type=float, default=0.3)
    parser.add_argument("--max-log-range-diff", type=float, default=None)
    args = parser.parse_args()
    if args.range_max <= args.range_min:
        parser.error("--range-max must be greater than --range-min")
    if args.boundary_log_thr < 0:
        parser.error("--boundary-log-thr must be nonnegative")
    if args.boundary_radius < 0:
        parser.error("--boundary-radius must be nonnegative")
    if not math.isfinite(args.correction_tol) or args.correction_tol < 0:
        parser.error("--correction-tol must be finite and nonnegative")
    if not args.gdc_label.strip():
        parser.error("--gdc-label must be non-empty")
    args.gdc_label = args.gdc_label.strip()
    if args.gdc_label in {"raw_sdn", "range_gdc"}:
        parser.error("--gdc-label must not collide with raw_sdn or range_gdc")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.sigma_angular <= 0 or args.sigma_tangent <= 0 or args.sigma_log_range <= 0:
        parser.error("graph sigma values must be positive")
    if args.max_log_range_diff is not None and args.max_log_range_diff <= 0:
        parser.error("--max-log-range-diff must be positive")
    return args


def main():
    analyze(parse_args())


if __name__ == "__main__":
    main()
