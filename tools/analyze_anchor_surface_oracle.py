#!/usr/bin/env python3
"""GT-assisted analysis of nearest-anchor sign mismatch and surface relation.

No correction or graph solve is performed.  GT range connectivity is used only
to determine whether a target cell and its nearest accepted anchor can be joined
without crossing a local range discontinuity.  Results are oracle diagnostics,
not inference-time measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import connected_components


REGIONS = ("overall", "boundary", "interior")
SCOPES = (
    "all_sign_eligible",
    "nearest_opposes_oracle",
    "wrong_direction_harm",
    "wrong_direction_harm_and_nearest_opposes",
)
RELATIONS = ("same_gt_component", "different_gt_component", "component_unavailable")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def valid_range(array: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.isfinite(array) & (array >= low) & (array <= high)


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


def read_split(path: str | Path) -> list[str]:
    with Path(path).open() as handle:
        ids = [f"{int(line.strip()):06d}" for line in handle if line.strip()]
    if not ids:
        raise ValueError(f"Empty split: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate frame ids: {path}")
    return sorted(ids)


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


def boundary_distance(gt: np.ndarray, valid: np.ndarray,
                      threshold: float) -> np.ndarray:
    log_gt = np.zeros(gt.shape, dtype=np.float64)
    log_gt[valid] = np.log(gt[valid])
    boundary = np.zeros(gt.shape, dtype=bool)
    horizontal = (
        valid & np.roll(valid, -1, axis=1)
        & (np.abs(log_gt - np.roll(log_gt, -1, axis=1)) > threshold)
    )
    boundary |= horizontal
    boundary |= np.roll(horizontal, 1, axis=1)
    if gt.shape[0] > 1:
        vertical = (
            valid[:-1] & valid[1:]
            & (np.abs(log_gt[:-1] - log_gt[1:]) > threshold)
        )
        boundary[:-1] |= vertical
        boundary[1:] |= vertical
    if not np.any(boundary):
        return np.full(gt.shape, np.inf, dtype=np.float64)
    width = gt.shape[1]
    tiled = np.tile(boundary, (1, 3))
    return distance_transform_edt(~tiled)[:, width:2 * width]


def gt_surface_components(gt: np.ndarray, valid: np.ndarray,
                          threshold: float) -> np.ndarray:
    """8-neighbor GT components after removing high log-range-difference edges."""
    node_rows, node_cols = np.where(valid)
    labels_image = np.full(gt.shape, -1, dtype=np.int32)
    if node_rows.size == 0:
        return labels_image
    node_id = np.full(gt.shape, -1, dtype=np.int32)
    node_id[node_rows, node_cols] = np.arange(node_rows.size, dtype=np.int32)
    log_gt = np.zeros(gt.shape, dtype=np.float64)
    log_gt[valid] = np.log(gt[valid])
    edge_i: list[np.ndarray] = []
    edge_j: list[np.ndarray] = []

    pair = (
        valid & np.roll(valid, -1, axis=1)
        & (np.abs(log_gt - np.roll(log_gt, -1, axis=1)) <= threshold)
    )
    rows, cols = np.where(pair)
    if rows.size:
        edge_i.append(node_id[rows, cols])
        edge_j.append(node_id[rows, (cols + 1) % gt.shape[1]])

    for dc in (-1, 0, 1):
        lower_valid = np.roll(valid[1:, :], -dc, axis=1)
        lower_log = np.roll(log_gt[1:, :], -dc, axis=1)
        pair = (
            valid[:-1, :] & lower_valid
            & (np.abs(log_gt[:-1, :] - lower_log) <= threshold)
        )
        rows, cols = np.where(pair)
        if rows.size:
            edge_i.append(node_id[rows, cols])
            edge_j.append(node_id[rows + 1, (cols + dc) % gt.shape[1]])

    if edge_i:
        first = np.concatenate(edge_i)
        second = np.concatenate(edge_j)
        graph = sparse.csr_matrix(
            (
                np.ones(first.size * 2, dtype=np.uint8),
                (np.concatenate((first, second)), np.concatenate((second, first))),
            ),
            shape=(node_rows.size, node_rows.size),
        )
    else:
        graph = sparse.csr_matrix((node_rows.size, node_rows.size), dtype=np.uint8)
    _, labels = connected_components(graph, directed=False, return_labels=True)
    labels_image[node_rows, node_cols] = labels.astype(np.int32)
    return labels_image


def nearest_periodic_anchor(anchor_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(anchor_mask):
        shape = anchor_mask.shape
        return (
            np.full(shape, np.inf, dtype=np.float64),
            np.full(shape, -1, dtype=np.int32),
            np.full(shape, -1, dtype=np.int32),
        )
    width = anchor_mask.shape[1]
    tiled = np.tile(anchor_mask, (1, 3))
    distances, indices = distance_transform_edt(~tiled, return_indices=True)
    center = slice(width, 2 * width)
    rows = indices[0][:, center].astype(np.int32)
    cols = (indices[1][:, center] % width).astype(np.int32)
    return distances[:, center], rows, cols


def sign(values: np.ndarray, tolerance: float) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.int8)
    output[values > tolerance] = 1
    output[values < -tolerance] = -1
    return output


def update(acc: dict[str, float], selected: np.ndarray,
           nearest_opposes: np.ndarray, harmed: np.ndarray,
           wrong_direction_harm: np.ndarray, error_change: np.ndarray) -> None:
    count = int(np.sum(selected))
    acc["pixels"] += count
    if not count:
        return
    acc["nearest_opposes"] += int(np.sum(selected & nearest_opposes))
    acc["harmed"] += int(np.sum(selected & harmed))
    acc["wrong_direction_harm"] += int(np.sum(selected & wrong_direction_harm))
    acc["total_harm"] += float(np.sum(np.maximum(error_change[selected], 0.0)))


def relation_masks(target_labels: np.ndarray, nearest_labels: np.ndarray) -> dict[str, np.ndarray]:
    available = (target_labels >= 0) & (nearest_labels >= 0)
    return {
        "same_gt_component": available & (target_labels == nearest_labels),
        "different_gt_component": available & (target_labels != nearest_labels),
        "component_unavailable": ~available,
    }


def bin_specs() -> tuple[tuple[str, tuple[float, ...], str], ...]:
    return (
        ("nearest_grid_distance", (0.0, 1.5, 2.5, 3.5, 4.5, 8.5, math.inf), "cells"),
        ("nearest_gt_log_difference", (0.0, 0.05, 0.10, 0.20, 0.40, math.inf), ""),
        ("nearest_pred_log_difference", (0.0, 0.05, 0.10, 0.20, 0.40, math.inf), ""),
    )


def validate_inputs(args: argparse.Namespace, ids: list[str]) -> dict[str, dict[str, str]]:
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
        missing = [frame for frame in ids if frame not in mapping]
        if missing:
            raise FileNotFoundError(
                f"{name} missing {len(missing)} split frames; sample={missing[:5]}"
            )
        mappings[name] = mapping
    return mappings


def analyze(args: argparse.Namespace) -> None:
    ids = read_split(args.split_file)
    if args.max_items is not None:
        ids = ids[:args.max_items]
    mappings = validate_inputs(args, ids)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filenames = (
        "anchor_surface_relation_summary.csv",
        "anchor_condition_summary.csv",
        "anchor_surface_metadata.json",
    )
    existing = [name for name in filenames if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Existing outputs {existing}; pass --overwrite")

    relation_acc = defaultdict(lambda: defaultdict(float))
    condition_acc = defaultdict(lambda: defaultdict(float))
    source_rows = np.asarray(args.source_rows, dtype=np.int64)

    for frame_index, frame in enumerate(ids, start=1):
        arrays = {
            name: np.load(mapping[frame]).astype(np.float64)
            for name, mapping in mappings.items()
        }
        shape = arrays["gt"].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
            raise ValueError(f"{frame}: inputs must be equal-shape 2D arrays")
        raw, gt, rgc, anchor = (
            arrays["raw"], arrays["gt"], arrays["rgc"], arrays["anchor"]
        )
        raw_valid = valid_range(raw, args.range_min, args.range_max)
        gt_valid = valid_range(gt, args.range_min, args.range_max)
        accepted = (
            raw_valid & valid_range(anchor, args.range_min, args.range_max)
            & (np.abs(anchor - raw) < args.anchor_reject_thr)
        )
        hidden = np.ones(shape, dtype=bool)
        hidden[source_rows, :] = False
        common = (
            hidden & raw_valid & gt_valid
            & valid_range(rgc, args.range_min, args.range_max)
        )
        for name in mappings:
            if name.startswith("support_"):
                common &= valid_range(arrays[name], args.range_min, args.range_max)

        nearest_distance, nearest_rows, nearest_cols = nearest_periodic_anchor(accepted)
        nearest_available = nearest_rows >= 0
        nearest_anchor_residual = np.full(shape, np.nan, dtype=np.float64)
        nearest_gt = np.full(shape, np.nan, dtype=np.float64)
        nearest_pred = np.full(shape, np.nan, dtype=np.float64)
        nearest_anchor_residual[nearest_available] = (
            anchor[nearest_rows[nearest_available], nearest_cols[nearest_available]]
            - raw[nearest_rows[nearest_available], nearest_cols[nearest_available]]
        )
        nearest_gt[nearest_available] = gt[
            nearest_rows[nearest_available], nearest_cols[nearest_available]
        ]
        nearest_pred[nearest_available] = raw[
            nearest_rows[nearest_available], nearest_cols[nearest_available]
        ]

        oracle_target = gt - raw
        correction = rgc - raw
        raw_abs = np.abs(oracle_target)
        rgc_abs = np.abs(correction - oracle_target)
        error_change = rgc_abs - raw_abs
        harmed = error_change > args.error_tol
        oracle_sign = sign(oracle_target, args.sign_tol)
        nearest_sign = sign(nearest_anchor_residual, args.sign_tol)
        correction_sign = sign(correction, args.sign_tol)
        sign_eligible = common & (oracle_sign != 0) & (nearest_sign != 0)
        nearest_opposes = sign_eligible & (nearest_sign == -oracle_sign)
        wrong_direction_harm = (
            common & harmed & (oracle_sign != 0)
            & (correction_sign == -oracle_sign)
        )

        distance = boundary_distance(gt, gt_valid, args.boundary_log_thr)
        regions = {
            "overall": common,
            "boundary": common & (distance <= args.boundary_radius),
            "interior": common & (distance >= 4.0),
        }
        scopes = {
            "all_sign_eligible": sign_eligible,
            "nearest_opposes_oracle": nearest_opposes,
            "wrong_direction_harm": wrong_direction_harm,
            "wrong_direction_harm_and_nearest_opposes": wrong_direction_harm & nearest_opposes,
        }

        for threshold in args.component_log_thresholds:
            components = gt_surface_components(gt, gt_valid, threshold)
            nearest_components = np.full(shape, -1, dtype=np.int32)
            nearest_components[nearest_available] = components[
                nearest_rows[nearest_available], nearest_cols[nearest_available]
            ]
            relations = relation_masks(components, nearest_components)
            for region, region_mask in regions.items():
                for scope, scope_mask in scopes.items():
                    for relation, relation_mask in relations.items():
                        selected = region_mask & scope_mask & relation_mask
                        update(
                            relation_acc[(threshold, region, scope, relation)],
                            selected, nearest_opposes, harmed,
                            wrong_direction_harm, error_change,
                        )

        gt_pair_valid = common & nearest_available & valid_range(
            nearest_gt, args.range_min, args.range_max
        )
        pred_pair_valid = common & nearest_available & valid_range(
            nearest_pred, args.range_min, args.range_max
        )
        gt_log_difference = np.full(shape, np.nan, dtype=np.float64)
        pred_log_difference = np.full(shape, np.nan, dtype=np.float64)
        gt_log_difference[gt_pair_valid] = np.abs(
            np.log(gt[gt_pair_valid]) - np.log(nearest_gt[gt_pair_valid])
        )
        pred_log_difference[pred_pair_valid] = np.abs(
            np.log(raw[pred_pair_valid]) - np.log(nearest_pred[pred_pair_valid])
        )
        feature_values = {
            "nearest_grid_distance": nearest_distance,
            "nearest_gt_log_difference": gt_log_difference,
            "nearest_pred_log_difference": pred_log_difference,
        }
        for region, region_mask in regions.items():
            for feature, edges, unit in bin_specs():
                values = feature_values[feature]
                for low, high in zip(edges[:-1], edges[1:]):
                    selected = (
                        region_mask & sign_eligible & np.isfinite(values)
                        & (values >= low) & (values < high)
                    )
                    update(
                        condition_acc[(region, feature, low, high, unit)],
                        selected, nearest_opposes, harmed,
                        wrong_direction_harm, error_change,
                    )

        if args.progress_every and (
            frame_index % args.progress_every == 0 or frame_index == len(ids)
        ):
            print(f"processed {frame_index}/{len(ids)}", flush=True)

    relation_rows = []
    group_totals = defaultdict(lambda: {"pixels": 0.0, "total_harm": 0.0})
    for (threshold, region, scope, _), acc in relation_acc.items():
        group = group_totals[(threshold, region, scope)]
        group["pixels"] += acc["pixels"]
        group["total_harm"] += acc["total_harm"]
    for (threshold, region, scope, relation), acc in sorted(relation_acc.items()):
        totals = group_totals[(threshold, region, scope)]
        relation_rows.append({
            "component_log_threshold": threshold,
            "region": region,
            "scope": scope,
            "surface_relation": relation,
            "pixels": int(acc["pixels"]),
            "pixel_share_within_scope": safe_ratio(acc["pixels"], totals["pixels"]),
            "nearest_opposes_ratio": safe_ratio(acc["nearest_opposes"], acc["pixels"]),
            "harm_rate": safe_ratio(acc["harmed"], acc["pixels"]),
            "wrong_direction_harm_rate": safe_ratio(acc["wrong_direction_harm"], acc["pixels"]),
            "total_harm": acc["total_harm"],
            "harm_share_within_scope": safe_ratio(acc["total_harm"], totals["total_harm"]),
        })

    condition_rows = []
    for (region, feature, low, high, unit), acc in sorted(condition_acc.items()):
        condition_rows.append({
            "region": region,
            "feature": feature,
            "bin_low_inclusive": low,
            "bin_high_exclusive": "inf" if math.isinf(high) else high,
            "bin_label": f"[{low:g}, {'inf' if math.isinf(high) else f'{high:g}'})",
            "unit": unit,
            "pixels": int(acc["pixels"]),
            "nearest_opposes_ratio": safe_ratio(acc["nearest_opposes"], acc["pixels"]),
            "harm_rate": safe_ratio(acc["harmed"], acc["pixels"]),
            "wrong_direction_harm_rate": safe_ratio(acc["wrong_direction_harm"], acc["pixels"]),
            "total_harm": acc["total_harm"],
            "mean_harm_over_bin": safe_ratio(acc["total_harm"], acc["pixels"]),
        })

    write_csv(output / "anchor_surface_relation_summary.csv", relation_rows)
    write_csv(output / "anchor_condition_summary.csv", condition_rows)
    metadata = {
        "analysis_kind": "gt_assisted_nearest_anchor_surface_oracle",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frames": len(ids),
        "split_file": str(Path(args.split_file).resolve()),
        "source_rows": args.source_rows,
        "component_log_thresholds": args.component_log_thresholds,
        "component_definition": "GT-valid 8-neighbor graph with edges retained when abs(log(R_i)-log(R_j)) <= threshold; horizontal indexing is periodic.",
        "nearest_anchor_definition": "Euclidean range-grid distance with periodic horizontal indexing; diagnostic proxy, not exact graph influence.",
        "gt_warning": "GT component and oracle residual signs are offline diagnostics and unavailable at inference.",
    }
    with (output / "anchor_surface_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Analyzed {len(ids)} frame(s)")
    print(f"Saved: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-range-dir", required=True)
    parser.add_argument("--gt-range-dir", required=True)
    parser.add_argument("--rgc-range-dir", required=True)
    parser.add_argument("--anchor-range-dir", required=True)
    parser.add_argument("--support-range-dir", action="append", default=[])
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rows", nargs="+", type=int, default=[5, 7, 9, 11])
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--anchor-reject-thr", type=float, default=2.0)
    parser.add_argument("--boundary-log-thr", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--component-log-thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--sign-tol", type=float, default=1e-3)
    parser.add_argument("--error-tol", type=float, default=1e-6)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.range_max <= args.range_min:
        parser.error("range-max must exceed range-min")
    if any(value < 0 or not math.isfinite(value) for value in args.component_log_thresholds):
        parser.error("component thresholds must be finite and nonnegative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("max-items must be positive")
    return args


if __name__ == "__main__":
    analyze(parse_args())
