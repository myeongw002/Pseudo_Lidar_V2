#!/usr/bin/env python3
"""Compare native GDC/RGC anchor rejection on identical LiDAR source points."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GDC_ROOT = REPO_ROOT / "gdc"
for path in (str(REPO_ROOT), str(GDC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data_utils.kitti_util import Calibration  # noqa: E402
from gdc import anchor_accept_mask, image_anchor_candidate_masks  # noqa: E402
from range_gdc.range_gdc import _apply_anchor_reject, valid_range_mask  # noqa: E402


CATEGORIES = ("both_accepted", "gdc_only_accepted", "rgc_only_accepted", "both_rejected")
DEFAULT_BINS = (0, 10, 20, 30, 40, 50, 60, 70, 80)


def read_ids(path):
    with open(path) as handle:
        return [f"{int(line.strip()):06d}" for line in handle if line.strip()]


def load_config(path):
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def load_points(path):
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path}: malformed KITTI point cloud")
    return values.reshape(-1, 4)


def source_locations(source_map, candidate_mask):
    """Map source IDs to their sole candidate representation coordinate."""
    source_map = np.asarray(source_map, dtype=np.int32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if source_map.shape != candidate_mask.shape:
        raise ValueError("source map and candidate mask shapes differ")
    locations = {}
    for row, col in zip(*np.where(candidate_mask & (source_map >= 0))):
        source_id = int(source_map[row, col])
        if source_id in locations:
            raise ValueError(f"source {source_id} occurs at multiple candidate locations")
        locations[source_id] = (int(row), int(col))
    return locations


def common_source_locations(
    gdc_source_map, gdc_candidate_mask, rgc_source_grid, rgc_candidate_mask,
):
    gdc = source_locations(gdc_source_map, gdc_candidate_mask)
    rgc = source_locations(rgc_source_grid, rgc_candidate_mask)
    return [(source_id, gdc[source_id], rgc[source_id]) for source_id in sorted(set(gdc) & set(rgc))]


def rejection_category(gdc_accept, rgc_accept):
    if gdc_accept and rgc_accept:
        return "both_accepted"
    if gdc_accept:
        return "gdc_only_accepted"
    if rgc_accept:
        return "rgc_only_accepted"
    return "both_rejected"


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)) if values.size else np.nan,
        f"{prefix}_median": float(np.median(values)) if values.size else np.nan,
        f"{prefix}_p90": float(np.percentile(values, 90)) if values.size else np.nan,
    }


def summarize_point_rows(rows, frame_id=None):
    counts = {category: sum(row["category"] == category for row in rows) for category in CATEGORIES}
    total = len(rows)
    mismatch = counts["gdc_only_accepted"] + counts["rgc_only_accepted"]
    result = {
        "total_common_candidates": total,
        **counts,
        "rejection_mismatch_count": mismatch,
        "rejection_mismatch_ratio": _ratio(mismatch, total),
        "gdc_accept_ratio": _ratio(counts["both_accepted"] + counts["gdc_only_accepted"], total),
        "rgc_accept_ratio": _ratio(counts["both_accepted"] + counts["rgc_only_accepted"], total),
        "both_accept_ratio": _ratio(counts["both_accepted"], total),
        "both_reject_ratio": _ratio(counts["both_rejected"], total),
        "gdc_only_accept_ratio": _ratio(counts["gdc_only_accepted"], total),
        "rgc_only_accept_ratio": _ratio(counts["rgc_only_accepted"], total),
    }
    result.update(_stats([row["gdc_abs_error"] for row in rows], "gdc_abs_error"))
    result.update(_stats([row["rgc_abs_error"] for row in rows], "rgc_abs_error"))
    for category, prefix in (("gdc_only_accepted", "gdc_only"), ("rgc_only_accepted", "rgc_only")):
        subset = [row for row in rows if row["category"] == category]
        result[f"{prefix}_mean_gdc_error"] = (
            float(np.mean([row["gdc_abs_error"] for row in subset])) if subset else np.nan
        )
        result[f"{prefix}_mean_rgc_error"] = (
            float(np.mean([row["rgc_abs_error"] for row in subset])) if subset else np.nan
        )
    if sum(counts.values()) != total:
        raise AssertionError("rejection category counts do not sum to common candidates")
    if frame_id is not None:
        result = {"frame_id": str(frame_id), **result}
    return result


def distance_rows(point_rows, bins=DEFAULT_BINS):
    output = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        subset = [row for row in point_rows if lower <= float(row["lidar_range"]) < upper]
        summary = summarize_point_rows(subset)
        output.append({
            "distance_min_m": float(lower), "distance_max_m": float(upper),
            "common_candidate_count": summary["total_common_candidates"],
            "gdc_accept_ratio": summary["gdc_accept_ratio"],
            "rgc_accept_ratio": summary["rgc_accept_ratio"],
            "rejection_mismatch_ratio": summary["rejection_mismatch_ratio"],
        })
    return output


def frame_point_rows(
    frame_id, points, canonical_source_grid, gdc_source_map, gdc_anchor,
    gdc_prediction, rgc_anchor, rgc_prediction, calib, selected_rows,
    reject_mode, abs_error_thr, log_ratio_thr, consider_range, range_min, range_max,
    gdc_subsample=False, gdc_subsample_strategy="deterministic", gdc_subsample_seed=None,
):
    selected_rows = np.asarray(selected_rows, dtype=np.int32)
    source_ids = np.unique(canonical_source_grid[selected_rows][canonical_source_grid[selected_rows] >= 0])
    source_ids.sort()
    if source_ids.size != points.shape[0]:
        raise ValueError(f"{frame_id}: canonical source count differs from shared PCD")
    point_by_source = {int(source_id): points[index] for index, source_id in enumerate(source_ids)}

    _, _, gdc_overlap, _ = image_anchor_candidate_masks(
        gdc_prediction, gdc_anchor, calib, consider_range=consider_range,
        subsample=gdc_subsample, subsample_strategy=gdc_subsample_strategy,
        subsample_seed=gdc_subsample_seed,
    )
    gdc_accepted = anchor_accept_mask(
        gdc_prediction, gdc_anchor, gdc_overlap, reject_mode, abs_error_thr, log_ratio_thr
    )
    rgc_overlap = (
        valid_range_mask(rgc_prediction, range_min, range_max)
        & valid_range_mask(rgc_anchor, range_min, range_max)
    )
    rgc_accepted, _ = _apply_anchor_reject(
        rgc_overlap, rgc_prediction, rgc_anchor, reject_mode, log_ratio_thr, abs_error_thr
    )
    common = common_source_locations(
        gdc_source_map, gdc_overlap, canonical_source_grid, rgc_overlap
    )
    rows = []
    for source_id, (gdc_row, gdc_col), (rgc_row, rgc_col) in common:
        point = point_by_source[source_id]
        camera_z = float(calib.project_velo_to_rect(point[None, :3])[0, 2])
        lidar_range = float(np.linalg.norm(point[:3].astype(np.float64)))
        gdc_pred = float(gdc_prediction[gdc_row, gdc_col])
        rgc_pred = float(rgc_prediction[rgc_row, rgc_col])
        gdc_error = abs(gdc_pred - camera_z)
        rgc_error = abs(rgc_pred - lidar_range)
        gdc_accept = bool(gdc_accepted[gdc_row, gdc_col])
        rgc_accept = bool(rgc_accepted[rgc_row, rgc_col])
        rows.append({
            "frame_id": str(frame_id), "source_index": source_id,
            "lidar_range": lidar_range, "lidar_camera_z": camera_z,
            "gdc_pred_z": gdc_pred, "rgc_pred_range": rgc_pred,
            "gdc_abs_error": gdc_error, "rgc_abs_error": rgc_error,
            "abs_error_difference": gdc_error - rgc_error,
            "gdc_accept": gdc_accept, "rgc_accept": rgc_accept,
            "category": rejection_category(gdc_accept, rgc_accept),
            "gdc_pixel_row": gdc_row, "gdc_pixel_col": gdc_col,
            "rgc_row": rgc_row, "rgc_col": rgc_col,
        })
    return rows


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--kitti-root", required=True)
    parser.add_argument("--config", default="configs/r64_pipeline_canonical.yaml")
    parser.add_argument("--sdn-depth-path", default=None)
    parser.add_argument("--rgc-pred-path", default=None)
    parser.add_argument("--distance-bins", default=",".join(str(v) for v in DEFAULT_BINS))
    parser.add_argument("--gdc-variant", choices=("naive", "optimized"), default="naive")
    return parser.parse_args()


def main():
    args = parse_args()
    root, kitti_root = Path(args.output_root), Path(args.kitti_root)
    split_file = Path(args.split_file)
    cfg = load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    selected_rows = cfg.get("range_anchor", {}).get("selected_rows", [5, 7, 9, 11])
    anchor_filter = cfg.get("anchor_filter", {})
    reject_mode = anchor_filter.get("mode", "abs")
    abs_error_thr = float(anchor_filter.get("abs_error_thr", 2.0))
    log_ratio_thr = float(anchor_filter.get("log_ratio_thr", 0.4))
    original = cfg.get("original_gdc", {})
    consider_range = original.get("consider_range", [-0.1, 3.0])
    gdc_subsample = args.gdc_variant == "optimized"
    gdc_subsample_strategy = original.get("subsample_strategy", "deterministic")
    gdc_subsample_seed = int(original.get("subsample_seed", 0))
    range_cfg = cfg.get("range_gdc", {})
    range_min, range_max = float(range_cfg.get("range_min", 0.1)), float(range_cfg.get("range_max", 80.0))
    data_tag = split_file.stem
    sdn_depth = Path(args.sdn_depth_path or root / "sdn" / "depth_maps" / data_tag)
    rgc_pred = Path(args.rgc_pred_path or root / "range" / "raw_sdn" / "G64_range")
    paths = {
        "points": root / "anchor" / "shared_canonical_pointcloud",
        "source": root / "anchor" / "shared_canonical_source_index",
        "gdc_source": root / "anchor" / "shared_canonical_image_source_index",
        "gdc_anchor": root / "anchor" / "shared_canonical_image_depth",
        "rgc_anchor": root / "anchor" / "range_shared_canonical" / "G64_range",
    }
    all_rows, frame_rows = [], []
    for frame_id in read_ids(split_file):
        points = load_points(paths["points"] / f"{frame_id}.bin")
        calib = Calibration(str(kitti_root / "calib" / f"{frame_id}.txt"))
        rows = frame_point_rows(
            frame_id, points,
            np.load(paths["source"] / f"{frame_id}.npy"),
            np.load(paths["gdc_source"] / f"{frame_id}.npy"),
            np.load(paths["gdc_anchor"] / f"{frame_id}.npy"),
            np.load(sdn_depth / f"{frame_id}.npy"),
            np.load(paths["rgc_anchor"] / f"{frame_id}.npy"),
            np.load(rgc_pred / f"{frame_id}.npy"),
            calib, selected_rows, reject_mode, abs_error_thr, log_ratio_thr,
            consider_range, range_min, range_max,
            gdc_subsample=gdc_subsample,
            gdc_subsample_strategy=gdc_subsample_strategy,
            gdc_subsample_seed=gdc_subsample_seed + int(frame_id),
        )
        all_rows.extend(rows)
        frame_rows.append(summarize_point_rows(rows, frame_id))
    summary = summarize_point_rows(all_rows)
    summary.update({
        "anchor_reject_mode": reject_mode, "abs_error_thr": abs_error_thr,
        "log_ratio_thr": log_ratio_thr, "consider_range_deg": json.dumps(consider_range),
        "gdc_variant": args.gdc_variant,
        "gdc_subsample_strategy": gdc_subsample_strategy if gdc_subsample else "disabled",
    })
    bins = [float(value) for value in args.distance_bins.split(",")]
    metrics = root / "metrics"
    write_csv(metrics / "anchor_rejection_consistency_per_point.csv", all_rows)
    write_csv(metrics / "anchor_rejection_consistency_per_frame.csv", frame_rows)
    write_csv(
        metrics / "anchor_rejection_consistency_summary.csv",
        [{"metric": key, "value": value} for key, value in summary.items()],
        ["metric", "value"],
    )
    write_csv(metrics / "anchor_rejection_consistency_distance.csv", distance_rows(all_rows, bins))
    print(f"common candidates: {summary['total_common_candidates']}")
    print(f"rejection mismatch: {summary['rejection_mismatch_count']} ({summary['rejection_mismatch_ratio']:.6f})")


if __name__ == "__main__":
    main()
