#!/usr/bin/env python3
"""Diagnose RGC wrong-direction harm and test inference-visible sign confidence.

The production RGC system is rebuilt with the same graph and objective. Positive
and negative accepted-anchor residuals are solved separately:

    A delta_pos = lambda_a D max(e, 0)
    A delta_neg = lambda_a D max(-e, 0)

Their difference reconstructs the original linear solution. Their normalized
disagreement defines an inference-visible confidence score. Ground truth is
used only to evaluate the score and to label wrong-direction harm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import cg


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.range_gdc import (  # noqa: E402
    _apply_anchor_reject,
    build_graph_residual_system,
    build_spherical_graph_laplacian,
    valid_range_mask,
)


REGIONS = ("overall", "boundary", "interior")
CONFIDENCE_EDGES = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.000001)
GATE_THRESHOLDS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def normalize_scene_id(path: str | Path) -> str:
    name = Path(path).stem
    patterns = (
        r"_G\d+_corr_range$", r"_G\d+_corr_mask$", r"_G\d+_range$",
        r"_G\d+_mask$", r"_R\d+_range$", r"_R\d+_mask$",
        r"_range$", r"_mask$",
    )
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", name)
            if updated != name:
                name = updated
                changed = True
    return name


def npy_map(directory: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.npy")):
        mapping.setdefault(normalize_scene_id(path), str(path))
    return mapping


def read_split(split_file: str | Path) -> list[str]:
    with Path(split_file).open() as handle:
        scene_ids = [f"{int(line.strip()):06d}" for line in handle if line.strip()]
    if not scene_ids:
        raise ValueError(f"Empty split: {split_file}")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"Duplicate ids in split: {split_file}")
    return sorted(scene_ids)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_projection(path: str | Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=True) as meta:
        required = ("vertical_centers_deg", "azimuth_centers_deg")
        missing = [key for key in required if key not in meta.files]
        if missing:
            raise ValueError(f"Projection metadata missing {missing}")
        return {
            "vertical_centers_deg": meta["vertical_centers_deg"].astype(np.float64),
            "azimuth_centers_deg": meta["azimuth_centers_deg"].astype(np.float64),
            "azimuth_mode": (
                str(meta["azimuth_mode"].item())
                if "azimuth_mode" in meta.files
                else "full_360_front_centered"
            ),
        }


def gt_boundary_distance(gt: np.ndarray, range_min: float, range_max: float,
                         threshold: float) -> np.ndarray:
    valid = valid_range_mask(gt, range_min, range_max)
    log_gt = np.zeros(gt.shape, dtype=np.float64)
    log_gt[valid] = np.log(gt[valid])
    boundary = np.zeros(gt.shape, dtype=bool)
    pair = (
        valid & np.roll(valid, -1, axis=1)
        & (np.abs(log_gt - np.roll(log_gt, -1, axis=1)) > threshold)
    )
    boundary |= pair
    boundary |= np.roll(pair, 1, axis=1)
    if gt.shape[0] > 1:
        pair = (
            valid[:-1] & valid[1:]
            & (np.abs(log_gt[:-1] - log_gt[1:]) > threshold)
        )
        boundary[:-1] |= pair
        boundary[1:] |= pair
    if not np.any(boundary):
        return np.full(gt.shape, np.inf, dtype=np.float64)
    width = gt.shape[1]
    tiled = np.tile(boundary, (1, 3))
    return distance_transform_edt(~tiled)[:, width:2 * width]


def nearest_periodic_anchor_residual(anchor_mask: np.ndarray,
                                     residual_map: np.ndarray) -> np.ndarray:
    if not np.any(anchor_mask):
        return np.full(anchor_mask.shape, np.nan, dtype=np.float64)
    width = anchor_mask.shape[1]
    tiled = np.tile(anchor_mask, (1, 3))
    _, indices = distance_transform_edt(~tiled, return_indices=True)
    rows = indices[0][:, width:2 * width]
    cols = indices[1][:, width:2 * width] % width
    return residual_map[rows, cols]


def solve_cg(A, b: np.ndarray) -> tuple[np.ndarray, int, float]:
    try:
        solution, info = cg(A, b, rtol=1e-6, atol=1e-8, maxiter=1000)
    except TypeError:
        solution, info = cg(A, b, tol=1e-6, maxiter=1000)
    residual = A @ solution - b
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-12))
    return solution, int(info), relative


def sign_with_tolerance(values: np.ndarray, tolerance: float) -> np.ndarray:
    signs = np.zeros(values.shape, dtype=np.int8)
    signs[values > tolerance] = 1
    signs[values < -tolerance] = -1
    return signs


def update_error_stats(acc: dict[str, float], raw_abs: np.ndarray,
                       method_abs: np.ndarray, wrong_direction: np.ndarray,
                       error_tol: float) -> None:
    change = method_abs - raw_abs
    improved = change < -error_tol
    harmed = change > error_tol
    acc["pixels"] += int(raw_abs.size)
    acc["raw_sum"] += float(np.sum(raw_abs))
    acc["method_sum"] += float(np.sum(method_abs))
    acc["improved"] += int(np.sum(improved))
    acc["harmed"] += int(np.sum(harmed))
    acc["wrong_direction"] += int(np.sum(wrong_direction))
    acc["total_improvement"] += float(np.sum(np.maximum(-change, 0.0)))
    acc["total_harm"] += float(np.sum(np.maximum(change, 0.0)))


def error_row(prefix: dict, acc: dict[str, float]) -> dict:
    pixels = int(acc["pixels"])
    return {
        **prefix,
        "pixels": pixels,
        "raw_mae": safe_ratio(acc["raw_sum"], pixels),
        "method_mae": safe_ratio(acc["method_sum"], pixels),
        "mae_improvement_vs_raw": safe_ratio(acc["raw_sum"] - acc["method_sum"], pixels),
        "improved_ratio": safe_ratio(acc["improved"], pixels),
        "harm_rate": safe_ratio(acc["harmed"], pixels),
        "wrong_direction_rate": safe_ratio(acc["wrong_direction"], pixels),
        "total_improvement": acc["total_improvement"],
        "total_harm": acc["total_harm"],
        "improvement_to_harm_ratio": safe_ratio(acc["total_improvement"], acc["total_harm"]),
    }


def validate_directories(args: argparse.Namespace,
                         scene_ids: list[str]) -> dict[str, dict[str, str]]:
    directories = {
        "raw": args.raw_range_dir,
        "gt": args.gt_range_dir,
        "rgc": args.rgc_range_dir,
        "anchor": args.anchor_range_dir,
    }
    for index, path in enumerate(args.support_range_dir):
        directories[f"support_{index}"] = path
    mappings = {}
    for name, directory in directories.items():
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)
        mapping = npy_map(directory)
        missing = [scene_id for scene_id in scene_ids if scene_id not in mapping]
        if missing:
            raise FileNotFoundError(
                f"{name} missing {len(missing)} split frames; sample={missing[:5]}"
            )
        mappings[name] = mapping
    return mappings


def analyze(args: argparse.Namespace) -> None:
    scene_ids = read_split(args.split_file)
    if args.max_items is not None:
        scene_ids = scene_ids[:args.max_items]
    mappings = validate_directories(args, scene_ids)
    projection = load_projection(args.projection_meta_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "confidence_bin_summary.csv", "confidence_gate_sweep.csv",
        "nearest_anchor_sign_summary.csv", "sign_solver_summary.csv",
        "sign_confidence_metadata.json",
    )
    existing = [name for name in output_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Existing outputs {existing}; pass --overwrite")

    source_rows = np.asarray(args.source_rows, dtype=np.int64)
    confidence_acc = defaultdict(lambda: defaultdict(float))
    gate_acc = defaultdict(lambda: defaultdict(float))
    nearest_acc = defaultdict(lambda: defaultdict(float))
    solver_rows = []

    for frame_index, scene_id in enumerate(scene_ids, start=1):
        arrays = {
            name: np.load(mapping[scene_id]).astype(np.float64)
            for name, mapping in mappings.items()
        }
        shape = arrays["gt"].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
            raise ValueError(f"{scene_id}: all inputs must be 2D with equal shape")
        if np.any(source_rows < 0) or np.any(source_rows >= shape[0]):
            raise ValueError(f"source rows outside frame shape {shape}")
        if projection["vertical_centers_deg"].shape != (shape[0],):
            raise ValueError("vertical projection metadata does not match range height")
        if projection["azimuth_centers_deg"].shape != (shape[1],):
            raise ValueError("azimuth projection metadata does not match range width")

        raw = arrays["raw"]
        gt = arrays["gt"]
        rgc = arrays["rgc"]
        anchor = arrays["anchor"]
        raw_valid = valid_range_mask(raw, args.range_min, args.range_max)
        anchor_valid = valid_range_mask(anchor, args.range_min, args.range_max)
        before_reject = raw_valid & anchor_valid
        accepted, _ = _apply_anchor_reject(
            before_reject, raw, anchor, "abs", 0.4, args.anchor_reject_thr
        )

        node_rows, node_cols = np.where(raw_valid)
        node_id = np.full(shape, -1, dtype=np.int32)
        node_id[node_rows, node_cols] = np.arange(node_rows.size, dtype=np.int32)
        L, _, _ = build_spherical_graph_laplacian(
            raw, raw_valid, node_id, node_rows, node_cols,
            vertical_centers_deg=projection["vertical_centers_deg"],
            azimuth_centers_deg=projection["azimuth_centers_deg"],
            azimuth_mode=projection["azimuth_mode"],
            neighbor="angular_grid8", edge_spatial_mode="angular",
            sigma_angular=args.sigma_angular, sigma_tangent=1.0,
            sigma_log_range=args.sigma_log_range,
            max_log_range_diff=None, edge_range_mode="log_gaussian",
        )
        target_indices = node_id[accepted]
        target_map = anchor - raw
        target_delta = target_map[accepted]
        positive_target = np.maximum(target_delta, 0.0)
        negative_target = np.maximum(-target_delta, 0.0)
        A, positive_rhs = build_graph_residual_system(
            L, target_indices, positive_target,
            lambda_anchor=args.lambda_anchor,
            lambda_smooth=args.lambda_smooth,
            lambda_prior=args.lambda_prior,
        )
        negative_rhs = np.zeros(node_rows.size, dtype=np.float64)
        negative_rhs[target_indices] = args.lambda_anchor * negative_target
        delta_positive, info_positive, residual_positive = solve_cg(A, positive_rhs)
        delta_negative, info_negative, residual_negative = solve_cg(A, negative_rhs)
        if info_positive != 0 or info_negative != 0:
            raise RuntimeError(
                f"{scene_id}: CG failed pos={info_positive}, neg={info_negative}"
            )

        positive_nonnegative = np.maximum(delta_positive, 0.0)
        negative_nonnegative = np.maximum(delta_negative, 0.0)
        reconstructed_delta = delta_positive - delta_negative
        confidence_nodes = np.abs(reconstructed_delta) / (
            positive_nonnegative + negative_nonnegative + args.confidence_eps
        )
        confidence_nodes = np.clip(confidence_nodes, 0.0, 1.0)
        confidence = np.full(shape, np.nan, dtype=np.float64)
        confidence[node_rows, node_cols] = confidence_nodes
        reconstructed = raw.copy()
        reconstructed[node_rows, node_cols] = np.clip(
            raw[node_rows, node_cols] + reconstructed_delta,
            args.range_min, args.range_max,
        )

        hidden = np.ones(shape, dtype=bool)
        hidden[source_rows, :] = False
        common = (
            hidden & raw_valid
            & valid_range_mask(gt, args.range_min, args.range_max)
            & valid_range_mask(rgc, args.range_min, args.range_max)
        )
        for name in mappings:
            if name.startswith("support_"):
                common &= valid_range_mask(arrays[name], args.range_min, args.range_max)
        boundary_distance = gt_boundary_distance(
            gt, args.range_min, args.range_max, args.boundary_log_thr
        )
        region_masks = {
            "overall": common,
            "boundary": common & (boundary_distance <= args.boundary_radius),
            "interior": common & (boundary_distance >= 4.0),
        }

        oracle_target = gt - raw
        existing_delta = rgc - raw
        raw_abs = np.abs(oracle_target)
        rgc_abs = np.abs(existing_delta - oracle_target)
        error_change = rgc_abs - raw_abs
        active = np.abs(existing_delta) > args.correction_tol
        wrong_direction = (
            active & (np.abs(oracle_target) > args.sign_tol)
            & (existing_delta * oracle_target < 0.0)
        )
        harmed_wrong_direction = wrong_direction & (error_change > args.error_tol)

        nearest_residual = nearest_periodic_anchor_residual(accepted, target_map)
        nearest_sign = sign_with_tolerance(nearest_residual, args.sign_tol)
        oracle_sign = sign_with_tolerance(oracle_target, args.sign_tol)
        nearest_relation = {
            "nearest_matches_oracle": (nearest_sign == oracle_sign) & (oracle_sign != 0),
            "nearest_opposes_oracle": (nearest_sign == -oracle_sign) & (oracle_sign != 0),
            "nearest_neutral": nearest_sign == 0,
        }

        identity_mask = common & np.isfinite(reconstructed)
        identity_diff = np.abs(reconstructed[identity_mask] - rgc[identity_mask])
        solver_rows.append({
            "frame_id": scene_id,
            "nodes": int(node_rows.size),
            "accepted_anchors": int(np.sum(accepted)),
            "positive_solver_info": info_positive,
            "negative_solver_info": info_negative,
            "positive_solve_residual": residual_positive,
            "negative_solve_residual": residual_negative,
            "reconstructed_vs_saved_max_abs": float(np.max(identity_diff)) if identity_diff.size else math.nan,
            "reconstructed_vs_saved_mean_abs": float(np.mean(identity_diff)) if identity_diff.size else math.nan,
        })

        for region, region_mask in region_masks.items():
            for low, high in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
                selected = (
                    region_mask & np.isfinite(confidence)
                    & (confidence >= low) & (confidence < high)
                )
                update_error_stats(
                    confidence_acc[(region, low, high)],
                    raw_abs[selected], rgc_abs[selected],
                    wrong_direction[selected], args.error_tol,
                )

            for threshold in GATE_THRESHOLDS:
                use_rgc = confidence >= threshold
                gated_delta = np.where(use_rgc, existing_delta, 0.0)
                gated_abs = np.abs(gated_delta - oracle_target)
                gated_wrong = (
                    (np.abs(gated_delta) > args.correction_tol)
                    & (np.abs(oracle_target) > args.sign_tol)
                    & (gated_delta * oracle_target < 0.0)
                )
                selected = region_mask
                update_error_stats(
                    gate_acc[(region, "hard_gate", threshold)],
                    raw_abs[selected], gated_abs[selected],
                    gated_wrong[selected], args.error_tol,
                )

            scaled_delta = np.nan_to_num(confidence, nan=0.0) * existing_delta
            scaled_abs = np.abs(scaled_delta - oracle_target)
            scaled_wrong = (
                (np.abs(scaled_delta) > args.correction_tol)
                & (np.abs(oracle_target) > args.sign_tol)
                & (scaled_delta * oracle_target < 0.0)
            )
            update_error_stats(
                gate_acc[(region, "continuous_scale", -1.0)],
                raw_abs[region_mask], scaled_abs[region_mask],
                scaled_wrong[region_mask], args.error_tol,
            )

            wrong_mask = region_mask & harmed_wrong_direction
            for relation, relation_mask in nearest_relation.items():
                selected = wrong_mask & relation_mask
                acc = nearest_acc[(region, relation)]
                acc["pixels"] += int(np.sum(selected))
                acc["total_harm"] += float(np.sum(error_change[selected]))
            unclassified = wrong_mask & ~np.logical_or.reduce(tuple(nearest_relation.values()))
            acc = nearest_acc[(region, "nearest_unclassified")]
            acc["pixels"] += int(np.sum(unclassified))
            acc["total_harm"] += float(np.sum(error_change[unclassified]))

        if args.progress_every and (
            frame_index % args.progress_every == 0 or frame_index == len(scene_ids)
        ):
            print(f"processed {frame_index}/{len(scene_ids)}", flush=True)

    confidence_rows = []
    for (region, low, high), acc in sorted(confidence_acc.items()):
        confidence_rows.append(error_row({
            "region": region,
            "confidence_low_inclusive": low,
            "confidence_high_exclusive": high,
            "confidence_bin": f"[{low:g}, {high:g})",
        }, acc))

    gate_rows = []
    for (region, mode, threshold), acc in sorted(gate_acc.items()):
        gate_rows.append(error_row({
            "region": region,
            "mode": mode,
            "confidence_threshold": "" if threshold < 0 else threshold,
        }, acc))

    nearest_rows = []
    for region in REGIONS:
        total_pixels = sum(
            int(acc["pixels"])
            for (candidate_region, _), acc in nearest_acc.items()
            if candidate_region == region
        )
        total_harm = sum(
            float(acc["total_harm"])
            for (candidate_region, _), acc in nearest_acc.items()
            if candidate_region == region
        )
        for (candidate_region, relation), acc in sorted(nearest_acc.items()):
            if candidate_region != region:
                continue
            nearest_rows.append({
                "region": region,
                "nearest_anchor_relation": relation,
                "wrong_direction_harmed_pixels": int(acc["pixels"]),
                "pixel_share_within_wrong_direction_harm": safe_ratio(acc["pixels"], total_pixels),
                "total_harm": acc["total_harm"],
                "harm_share_within_wrong_direction_harm": safe_ratio(acc["total_harm"], total_harm),
            })

    write_csv(output_dir / "confidence_bin_summary.csv", confidence_rows)
    write_csv(output_dir / "confidence_gate_sweep.csv", gate_rows)
    write_csv(output_dir / "nearest_anchor_sign_summary.csv", nearest_rows)
    write_csv(output_dir / "sign_solver_summary.csv", solver_rows)
    metadata = {
        "analysis_kind": "sign_confidence_diagnostic_and_non_gt_gate_ablation",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frames": len(scene_ids),
        "split_file": str(Path(args.split_file).resolve()),
        "source_rows": args.source_rows,
        "parameters": {
            "lambda_anchor": args.lambda_anchor,
            "lambda_smooth": args.lambda_smooth,
            "lambda_prior": args.lambda_prior,
            "sigma_angular": args.sigma_angular,
            "sigma_log_range": args.sigma_log_range,
            "anchor_reject_thr": args.anchor_reject_thr,
            "correction_tol": args.correction_tol,
            "error_tol": args.error_tol,
            "sign_tol": args.sign_tol,
            "confidence_eps": args.confidence_eps,
        },
        "confidence_definition": "abs(delta_positive-delta_negative)/(delta_positive+delta_negative+eps)",
        "gt_usage": "GT is used only for region labels and outcome evaluation; confidence and gates use Raw plus accepted anchors only.",
    }
    with (output_dir / "sign_confidence_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    max_identity = max(
        (row["reconstructed_vs_saved_max_abs"] for row in solver_rows),
        default=math.nan,
    )
    print(f"Analyzed {len(scene_ids)} frame(s)")
    print(f"max reconstructed-vs-saved difference: {max_identity:.9g} m")
    print(f"Saved: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-range-dir", required=True)
    parser.add_argument("--gt-range-dir", required=True)
    parser.add_argument("--rgc-range-dir", required=True)
    parser.add_argument("--anchor-range-dir", required=True)
    parser.add_argument("--support-range-dir", action="append", default=[])
    parser.add_argument("--projection-meta-path", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rows", nargs="+", type=int, default=[5, 7, 9, 11])
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--anchor-reject-thr", type=float, default=2.0)
    parser.add_argument("--lambda-anchor", type=float, default=100.0)
    parser.add_argument("--lambda-smooth", type=float, default=1.0)
    parser.add_argument("--lambda-prior", type=float, default=0.1)
    parser.add_argument("--sigma-angular", type=float, default=0.01)
    parser.add_argument("--sigma-log-range", type=float, default=0.3)
    parser.add_argument("--boundary-log-thr", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--correction-tol", type=float, default=1e-3)
    parser.add_argument("--error-tol", type=float, default=1e-6)
    parser.add_argument("--sign-tol", type=float, default=1e-3)
    parser.add_argument("--confidence-eps", type=float, default=1e-12)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.range_max <= args.range_min:
        parser.error("range-max must exceed range-min")
    for field in (
        "anchor_reject_thr", "lambda_anchor", "lambda_smooth", "lambda_prior",
        "sigma_angular", "sigma_log_range", "boundary_log_thr",
        "correction_tol", "error_tol", "sign_tol", "confidence_eps",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{field.replace('_', '-')} must be finite and nonnegative")
    if args.sigma_angular == 0 or args.sigma_log_range == 0:
        parser.error("sigma values must be positive")
    if args.boundary_radius < 0:
        parser.error("boundary-radius must be nonnegative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("max-items must be positive")
    if args.progress_every < 0:
        parser.error("progress-every must be nonnegative")
    return args


if __name__ == "__main__":
    analyze(parse_args())
