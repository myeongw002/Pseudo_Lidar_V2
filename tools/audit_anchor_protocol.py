#!/usr/bin/env python3
"""Quantify differences between Original-GDC and fixed-row Range-GDC anchors."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from range_gdc.range_projection import find_input_npy
from src.pseudo_lidar.depth_to_range_uniform import lidar_points_to_spherical_guide_uniform


FRAME_FIELDS = (
    "scene_id", "gdc_sparse_physical_point_count", "gdc_projected_valid_cell_count",
    "rgc_fixed_row_valid_cell_count", "overlapping_grid_cell_count",
    "overlap_over_gdc_projected_cells", "overlap_over_rgc_anchor_cells",
    "overlap_range_mae", "overlap_range_median", "overlap_range_p95",
    "gdc_projected_outside_selected_rows_count", "gdc_projected_outside_selected_rows_ratio",
    "rgc_anchor_not_from_gdc_count", "rgc_anchor_not_from_gdc_ratio",
)


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_split_ids(path):
    with open(path) as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def load_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def scalar(meta, name, default):
    if name not in meta.files:
        return default
    value = meta[name]
    return value.item() if getattr(value, "ndim", 0) == 0 else value


def projection_kwargs(meta):
    return {
        "range_h": int(scalar(meta, "height", 64)),
        "range_w": int(scalar(meta, "width", 1024)),
        "vmin_deg": float(scalar(meta, "vmin_deg", -24.9)),
        "vmax_deg": float(scalar(meta, "vmax_deg", 2.0)),
        "azimuth_mode": str(scalar(meta, "azimuth_mode", "full_360_front_centered")),
        "azimuth_min_deg": scalar(meta, "azimuth_min_deg", None),
        "azimuth_max_deg": scalar(meta, "azimuth_max_deg", None),
        "range_min": float(scalar(meta, "depth_min", 0.1)),
        "range_max": float(scalar(meta, "depth_max", 80.0)),
        "invalid_value": float(scalar(meta, "invalid_value", 0.0)),
    }


def project_gdc_sparse_points(points, kwargs):
    if points.size == 0:
        return np.full((kwargs["range_h"], kwargs["range_w"]), kwargs["invalid_value"], dtype=np.float32)
    grid, _ = lidar_points_to_spherical_guide_uniform(points[:, :3], **kwargs)
    return grid.astype(np.float32)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def frame_row(scene_id, sparse_path, anchor_path, kwargs, selected_rows):
    values = np.fromfile(sparse_path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{sparse_path}: expected float32 x/y/z/intensity records")
    points = values.reshape(-1, 4)
    projected = project_gdc_sparse_points(points, kwargs)
    anchor = np.load(find_input_npy(str(anchor_path), scene_id)).astype(np.float32)
    if anchor.shape != projected.shape:
        raise ValueError(f"{scene_id:06d}: anchor shape {anchor.shape} != projection shape {projected.shape}")
    projected_valid = np.isfinite(projected) & (projected > kwargs["invalid_value"])
    anchor_valid = np.isfinite(anchor) & (anchor > kwargs["invalid_value"])
    overlap = projected_valid & anchor_valid
    selected_mask = np.zeros(projected.shape[0], dtype=bool)
    selected_mask[selected_rows] = True
    outside = projected_valid & ~selected_mask[:, None]
    abs_difference = np.abs(projected[overlap].astype(np.float64) - anchor[overlap].astype(np.float64))
    gdc_count = int(projected_valid.sum())
    rgc_count = int(anchor_valid.sum())
    overlap_count = int(overlap.sum())
    not_from_gdc = anchor_valid & ~projected_valid
    return {
        "scene_id": f"{scene_id:06d}",
        "gdc_sparse_physical_point_count": int(points.shape[0]),
        "gdc_projected_valid_cell_count": gdc_count,
        "rgc_fixed_row_valid_cell_count": rgc_count,
        "overlapping_grid_cell_count": overlap_count,
        "overlap_over_gdc_projected_cells": safe_ratio(overlap_count, gdc_count),
        "overlap_over_rgc_anchor_cells": safe_ratio(overlap_count, rgc_count),
        "overlap_range_mae": float(np.mean(abs_difference)) if abs_difference.size else np.nan,
        "overlap_range_median": float(np.median(abs_difference)) if abs_difference.size else np.nan,
        "overlap_range_p95": float(np.percentile(abs_difference, 95)) if abs_difference.size else np.nan,
        "gdc_projected_outside_selected_rows_count": int(outside.sum()),
        "gdc_projected_outside_selected_rows_ratio": safe_ratio(int(outside.sum()), gdc_count),
        "rgc_anchor_not_from_gdc_count": int(not_from_gdc.sum()),
        "rgc_anchor_not_from_gdc_ratio": safe_ratio(int(not_from_gdc.sum()), rgc_count),
        "_overlap_abs_differences": abs_difference,
    }


def aggregate(rows):
    total = lambda key: int(sum(int(row[key]) for row in rows))
    differences = np.concatenate([row["_overlap_abs_differences"] for row in rows]) if rows else np.array([])
    gdc_cells = total("gdc_projected_valid_cell_count")
    rgc_cells = total("rgc_fixed_row_valid_cell_count")
    overlap = total("overlapping_grid_cell_count")
    outside = total("gdc_projected_outside_selected_rows_count")
    not_from_gdc = total("rgc_anchor_not_from_gdc_count")
    return [
        {"metric": "frame_count", "value": len(rows)},
        {"metric": "gdc_sparse_physical_point_count_total", "value": total("gdc_sparse_physical_point_count")},
        {"metric": "gdc_projected_valid_cell_count_total", "value": gdc_cells},
        {"metric": "rgc_fixed_row_valid_cell_count_total", "value": rgc_cells},
        {"metric": "overlapping_grid_cell_count_total", "value": overlap},
        {"metric": "overlap_over_gdc_projected_cells", "value": safe_ratio(overlap, gdc_cells)},
        {"metric": "overlap_over_rgc_anchor_cells", "value": safe_ratio(overlap, rgc_cells)},
        {"metric": "overlap_range_mae", "value": float(np.mean(differences)) if differences.size else np.nan},
        {"metric": "overlap_range_median", "value": float(np.median(differences)) if differences.size else np.nan},
        {"metric": "overlap_range_p95", "value": float(np.percentile(differences, 95)) if differences.size else np.nan},
        {"metric": "gdc_projected_outside_selected_rows_ratio", "value": safe_ratio(outside, gdc_cells)},
        {"metric": "rgc_anchor_not_from_gdc_ratio", "value": safe_ratio(not_from_gdc, rgc_cells)},
    ]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/r64_pipeline_test_1000.yaml")
    parser.add_argument("--pipeline-output-root", default=None)
    parser.add_argument("--gdc-sparse-path", default=None)
    parser.add_argument("--rgc-anchor-path", default=None)
    parser.add_argument("--projection-meta-path", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selected-rows", type=int, nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(resolve(args.config))
    root = Path(args.pipeline_output_root or cfg["output_root"]).expanduser()
    sparse_path = Path(args.gdc_sparse_path).expanduser() if args.gdc_sparse_path else root / "anchor" / "shared_4beam_pointcloud"
    anchor_path = Path(args.rgc_anchor_path).expanduser() if args.rgc_anchor_path else root / "anchor" / "range" / "G64_range"
    projection_meta_path = Path(args.projection_meta_path).expanduser() if args.projection_meta_path else root / "range" / "gt" / "meta" / "projection_meta.npz"
    split_file = resolve(args.split_file or cfg["split_file"])
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else root / "metrics"
    for path, label in ((sparse_path, "GDC sparse point cloud directory"), (anchor_path, "RGC anchor directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not projection_meta_path.is_file() or not split_file.is_file():
        raise FileNotFoundError("Projection metadata or split file is missing")

    meta = np.load(projection_meta_path, allow_pickle=True)
    kwargs = projection_kwargs(meta)
    selected_rows = np.asarray(
        args.selected_rows or cfg.get("range_anchor", {}).get("selected_rows", [5, 7, 9, 11]),
        dtype=np.int32,
    )
    if np.any(selected_rows < 0) or np.any(selected_rows >= kwargs["range_h"]):
        raise ValueError("--selected-rows are outside the spherical grid height")
    rows = []
    for scene_id in read_split_ids(split_file):
        sparse_file = sparse_path / f"{scene_id:06d}.bin"
        if not sparse_file.is_file():
            raise FileNotFoundError(f"Missing GDC sparse point cloud: {sparse_file}")
        rows.append(frame_row(scene_id, sparse_file, anchor_path, kwargs, selected_rows))

    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    audit_path = output_dir / "anchor_protocol_audit.csv"
    summary_path = output_dir / "anchor_protocol_summary.csv"
    write_csv(audit_path, FRAME_FIELDS, public_rows)
    write_csv(summary_path, ("metric", "value"), aggregate(rows))
    print(f"Saved per-frame anchor protocol audit: {audit_path}")
    print(f"Saved aggregate anchor protocol summary: {summary_path}")


if __name__ == "__main__":
    main()
