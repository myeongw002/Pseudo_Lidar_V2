#!/usr/bin/env python3
"""Write unified per-frame and summary range-image evaluation CSV files."""

import argparse
import csv
import math
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .range_projection import find_input_npy
except ImportError:
    from range_projection import find_input_npy


METRICS_FIELDNAMES = [
    "schema_version",
    "name",
    "method",
    "area",
    "area_legacy",
    "mae",
    "rmse",
    "median_abs",
    "p90_abs",
    "p95_abs",
    "p99_abs",
    "max_abs",
    "mean_error",
    "std_error",
    "eval_pixels",
    "gt_valid_pixels",
    "eval_ratio_over_gt",
    "area_gt_valid_pixels",
    "area_pred_valid_pixels",
    "area_coverage",
    "coverage_in_area",
    "missing_prediction_pixels",
    "eval_domain",
    "sigma_guide",
    "sigma_row",
    "sigma_col",
    "col_radius",
    "row_offset",
    "row_stride",
    "low_h",
    "target_h",
    "missing_gt_pixels",
    "A_low_valid_guide_valid_count",
    "A_low_valid_guide_valid_ratio",
    "B_low_valid_guide_invalid_count",
    "B_low_valid_guide_invalid_ratio",
    "C_low_invalid_guide_valid_count",
    "C_low_invalid_guide_valid_ratio",
    "D_low_invalid_guide_invalid_count",
    "D_low_invalid_guide_invalid_ratio",
]

SUMMARY_FIELDNAMES = [
    "schema_version",
    "eval_domain",
    "method",
    "area",
    "area_legacy",
    "frames",
    "frames_total",
    "frames_with_eval",
    "mae_mean",
    "mae_weighted",
    "rmse_mean",
    "rmse_weighted",
    "median_abs_mean",
    "p90_abs_mean",
    "p95_abs_mean",
    "p99_abs_mean",
    "max_abs_mean",
    "mean_error_mean",
    "std_error_mean",
    "eval_pixels",
    "area_gt_valid_pixels",
    "area_pred_valid_pixels",
    "area_coverage",
    "coverage_in_area",
    "missing_prediction_pixels",
]

LEAKAGE_FIELDNAMES = [
    "schema_version",
    "name",
    "status",
    "warning",
    "danger",
    "target_rows",
    "source_rows",
    "expected_anchor_rows",
    "anchor_valid_rows",
    "anchor_valid_rows_ratio_to_source",
    "anchor_path_looks_like_source_rows",
    "anchor_valid_pixels",
    "anchor_valid_pixels_in_anchor_rows",
    "anchor_valid_pixels_in_hidden_rows",
    "anchor_hidden_valid_ratio",
    "force_anchor_hidden_possible",
    "hidden_gt_valid_pixels",
    "hidden_eval_pixels",
    "hidden_mae",
    "hidden_rmse",
    "hidden_p90",
    "hidden_p95",
    "hidden_p99",
    "hidden_err_zero_ratio",
    "hidden_err_5cm_ratio",
    "warn_err_zero_ratio",
    "warn_err_5cm_ratio",
]

LEAKAGE_SUMMARY_FIELDNAMES = [
    "schema_version",
    "leakage_method",
    "frames_total",
    "frames_with_warning",
    "frames_with_danger",
    "mean_num_anchor_rows",
    "max_num_anchor_rows",
    "sum_anchor_valid_in_hidden_rows",
    "mean_hidden_err_lt_1e-6_ratio",
    "mean_hidden_err_lt_5e-2_ratio",
    "weighted_hidden_mae",
    "weighted_hidden_rmse",
    "weighted_hidden_p90",
    "weighted_hidden_p95",
    "weighted_hidden_p99",
    "status",
]

DISTANCE_METRICS_FIELDNAMES = [
    "schema_version",
    "frame",
    "area",
    "distance_bin",
    "bin_min",
    "bin_max",
    "method",
    "mae",
    "rmse",
    "median_abs",
    "p90_abs",
    "p95_abs",
    "p99_abs",
    "max_abs",
    "mean_error_signed",
    "std_abs",
    "area_gt_valid_pixels",
    "eval_pixels",
    "missing_prediction_pixels",
    "coverage_in_area",
]

DISTANCE_SUMMARY_FIELDNAMES = [
    "schema_version",
    "area",
    "distance_bin",
    "bin_min",
    "bin_max",
    "method",
    "frames",
    "area_gt_valid_pixels",
    "eval_pixels",
    "missing_prediction_pixels",
    "coverage_in_area",
    "mae",
    "rmse",
    "median_abs",
    "p90_abs",
    "p95_abs",
    "p99_abs",
    "max_abs",
    "mean_error_signed",
    "std_abs",
    "delta_mae_vs_raw",
    "rel_mae_improve_vs_raw",
    "delta_median_vs_raw",
    "rel_median_improve_vs_raw",
    "delta_p90_vs_raw",
    "rel_p90_improve_vs_raw",
    "delta_p95_vs_raw",
    "rel_p95_improve_vs_raw",
]

AREAS = [
    "full_valid",
    "anchor_rows_valid",
    "hidden_rows_valid",
    "common_full_valid",
    "common_hidden_valid",
]

DISTANCE_AREAS = [
    "hidden_rows_valid",
    "common_hidden_valid",
    "full_valid",
]

SCHEMA_VERSION = "unified_eval_v1"


def read_split_ids(split_file):
    with open(split_file) as f:
        return [int(x.strip()) for x in f.readlines() if x.strip()]


def valid_range_mask(range_img, range_min, range_max):
    return (
        np.isfinite(range_img)
        & (range_img >= float(range_min))
        & (range_img <= float(range_max))
    )


def parse_method(value):
    if "=" not in value:
        raise ValueError(f"--method must be name=path, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"--method must be name=path, got {value!r}")
    return name, path


def safe_mean(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return math.nan
    return float(np.mean(values))


def parse_distance_bins(value):
    if isinstance(value, str):
        edges = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        edges = [float(item) for item in value]
    if len(edges) < 2:
        raise ValueError("--distance_bins must contain at least two edges")
    if any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("--distance_bins must be strictly increasing")
    return [(edges[idx], edges[idx + 1]) for idx in range(len(edges) - 1)]


def distance_bin_label(bin_min, bin_max):
    def fmt(value):
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    return f"{fmt(bin_min)}-{fmt(bin_max)}m"


def nan_distance_metrics():
    return {
        "mae": math.nan,
        "rmse": math.nan,
        "median_abs": math.nan,
        "p90_abs": math.nan,
        "p95_abs": math.nan,
        "p99_abs": math.nan,
        "max_abs": math.nan,
        "mean_error_signed": math.nan,
        "std_abs": math.nan,
    }


def error_metrics(errors):
    if errors.size == 0:
        return nan_distance_metrics()
    abs_errors = np.abs(errors)
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "median_abs": float(np.median(abs_errors)),
        "p90_abs": float(np.percentile(abs_errors, 90)),
        "p95_abs": float(np.percentile(abs_errors, 95)),
        "p99_abs": float(np.percentile(abs_errors, 99)),
        "max_abs": float(np.max(abs_errors)),
        "mean_error_signed": float(np.mean(errors)),
        "std_abs": float(np.std(abs_errors)),
    }


def load_meta_selected_rows(projection_meta_path):
    if not projection_meta_path:
        return None
    if not osp.exists(projection_meta_path):
        return None
    meta = np.load(projection_meta_path, allow_pickle=True)
    if "selected_rows" not in meta.files:
        return None
    return meta["selected_rows"].astype(np.int32)


def load_projection_meta(projection_meta_path):
    if not projection_meta_path or not osp.exists(projection_meta_path):
        return None
    return np.load(projection_meta_path, allow_pickle=True)


def scalar_meta(meta, key, default=None):
    if meta is None or key not in meta.files:
        return default
    value = meta[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def frame_projection_meta(input_path, scene_id):
    path = Path(input_path)
    root = path.parent if path.name in {"G64_range", "G64_mask"} else path
    candidate = root / "meta" / f"{int(scene_id):06d}_guide_projection_meta.npz"
    return np.load(candidate, allow_pickle=True) if candidate.exists() else None


def validate_matching_projection_metadata(scene_id, named_paths):
    """Fail when available GT/prediction frame metadata describe different grids."""
    keys = (
        "height", "width", "vmin_deg", "vmax_deg", "azimuth_mode",
        "azimuth_min_deg", "azimuth_max_deg", "row0_convention",
        "vertical_centers_deg", "azimuth_centers_deg",
    )
    loaded = [(name, frame_projection_meta(path, scene_id)) for name, path in named_paths]
    loaded = [(name, meta) for name, meta in loaded if meta is not None]
    if len(loaded) < 2:
        return
    base_name, base = loaded[0]
    for name, meta in loaded[1:]:
        for key in keys:
            if key not in base.files or key not in meta.files:
                continue
            if not np.array_equal(base[key], meta[key]):
                raise ValueError(
                    f"{scene_id:06d}: projection metadata mismatch for {key}: "
                    f"{base_name} vs {name}"
                )


def validate_projection_meta_once(meta, gt, guide, pred_by_method, args):
    expected_h = int(args.expected_height)
    expected_w = int(args.expected_width)
    if meta is not None:
        meta_h = int(scalar_meta(meta, "range_h", scalar_meta(meta, "height", expected_h)))
        meta_w = int(scalar_meta(meta, "range_w", scalar_meta(meta, "width", expected_w)))
        if (meta_h, meta_w) != (expected_h, expected_w):
            raise ValueError(
                f"Projection meta shape mismatch: expected {(expected_h, expected_w)}, "
                f"got {(meta_h, meta_w)}"
            )
    expected_shape = (expected_h, expected_w)

    errors = []
    if gt.shape != expected_shape:
        errors.append(f"GT R64 shape {gt.shape} != expected shape {expected_shape}")
    if guide.shape != expected_shape:
        errors.append(f"anchor/guide shape {guide.shape} != expected shape {expected_shape}")
    for method, pred in pred_by_method.items():
        if pred.shape != expected_shape:
            errors.append(f"{method} shape {pred.shape} != expected shape {expected_shape}")

    if meta is not None:
        azimuth_mode = str(scalar_meta(meta, "azimuth_mode", ""))
        invalid_value = float(scalar_meta(meta, "invalid_value", np.nan))
        vmin = float(scalar_meta(meta, "vmin_deg", np.nan))
        vmax = float(scalar_meta(meta, "vmax_deg", np.nan))
        row0 = str(scalar_meta(meta, "row0_convention", ""))
        if azimuth_mode not in {"front_center", "full_360_front_centered", "bounded"}:
            errors.append(f"unsupported azimuth_mode {azimuth_mode!r}")
        if np.isfinite(invalid_value) and invalid_value != 0.0:
            errors.append(f"invalid_value is {invalid_value}, expected 0")
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax <= vmin:
            errors.append(f"invalid vertical bounds: vmin_deg={vmin}, vmax_deg={vmax}")
        if row0 and row0 != "top_highest_elevation":
            errors.append(f"row0_convention is {row0!r}, expected 'top_highest_elevation'")

    if errors:
        raise ValueError("Projection metadata/grid mismatch:\n  - " + "\n  - ".join(errors))


def frame_selected_rows(selected_rows_dir, scene_id):
    if not selected_rows_dir:
        return None
    path = osp.join(selected_rows_dir, f"{int(scene_id):06d}_selected_rows.npy")
    if not osp.exists(path):
        return None
    return np.load(path).astype(np.int32)


def selected_rows(
    target_h,
    source_rows,
    row_offset=None,
    row_stride=None,
    source_row_indices=None,
    scene_id=None,
    selected_rows_dir=None,
    projection_selected_rows=None,
):
    rows = frame_selected_rows(selected_rows_dir, scene_id) if scene_id is not None else None
    if rows is None and projection_selected_rows is not None:
        rows = np.asarray(projection_selected_rows, dtype=np.int32)
    if rows is None and source_row_indices is not None:
        rows = np.asarray(source_row_indices, dtype=np.int32)
    if rows is not None:
        if rows.size == 0:
            raise ValueError("selected rows must contain at least one row")
        if np.any(rows < 0) or np.any(rows >= int(target_h)):
            raise ValueError(f"selected rows contain rows outside target height {target_h}")
        return np.unique(rows)
    if row_stride is not None:
        return np.arange(int(row_offset or 0), int(target_h), int(row_stride), dtype=np.int32)
    if source_rows is None:
        raise ValueError("Either --source_rows or --row_stride must be provided")
    if int(source_rows) <= 0 or int(target_h) <= 0:
        raise ValueError("--source_rows and --target_h must be positive")
    if int(source_rows) >= int(target_h):
        return np.arange(int(target_h), dtype=np.int32)
    centers = (np.arange(int(source_rows), dtype=np.float64) + 0.5) * (
        float(target_h) / float(source_rows)
    )
    rows = np.floor(centers).astype(np.int32)
    return np.unique(np.clip(rows, 0, int(target_h) - 1))


def row_masks(shape, args, scene_id=None):
    rows = selected_rows(
        shape[0],
        args.source_rows,
        row_offset=args.row_offset,
        row_stride=args.row_stride,
        source_row_indices=args.source_row_indices,
        scene_id=scene_id,
        selected_rows_dir=args.selected_rows_dir,
        projection_selected_rows=args.projection_selected_rows,
    )
    anchor = np.zeros(shape, dtype=bool)
    anchor[rows, :] = True
    return anchor, ~anchor


def anchor_valid_rows(anchor_valid):
    return np.where(anchor_valid.any(axis=1))[0]


def leakage_row(scene_id, pred_by_method, gt, anchor, args):
    if not args.enable_leakage_check:
        return None, None
    if args.leakage_method not in pred_by_method:
        raise ValueError(
            f"--leakage_method {args.leakage_method!r} is not among methods: "
            f"{sorted(pred_by_method)}"
        )

    gt_valid = valid_range_mask(gt, args.range_min, args.range_max)
    anchor_valid = valid_range_mask(anchor, args.range_min, args.range_max)
    anchor_rows_mask, hidden_rows_mask = row_masks(gt.shape, args, scene_id)
    pred = pred_by_method[args.leakage_method]
    pred_valid = valid_range_mask(pred, args.range_min, args.range_max)

    rows = anchor_valid_rows(anchor_valid)
    expected_rows = selected_rows(
        gt.shape[0],
        args.source_rows,
        row_offset=args.row_offset,
        row_stride=args.row_stride,
        source_row_indices=args.source_row_indices,
        scene_id=scene_id,
        selected_rows_dir=args.selected_rows_dir,
        projection_selected_rows=args.projection_selected_rows,
    )
    expected_row_count = int(len(expected_rows))
    if gt.shape != (int(args.expected_height), int(args.expected_width)):
        raise ValueError(f"{scene_id:06d}: gt_range shape mismatch: expected {(args.expected_height, args.expected_width)}, got {gt.shape}")
    if anchor.shape != (int(args.expected_height), int(args.expected_width)):
        raise ValueError(f"{scene_id:06d}: anchor_range shape mismatch: expected {(args.expected_height, args.expected_width)}, got {anchor.shape}")
    if pred.shape != (int(args.expected_height), int(args.expected_width)):
        raise ValueError(f"{scene_id:06d}: pred_range shape mismatch: expected {(args.expected_height, args.expected_width)}, got {pred.shape}")
    unexpected_rows = np.setdiff1d(rows, expected_rows)
    if unexpected_rows.size:
        raise ValueError(
            f"{scene_id:06d}: Anchor contains rows outside the fixed evaluation rows: "
            f"expected={expected_rows}, actual={rows}, unexpected={unexpected_rows}"
        )
    # A configured row may contain no valid GT point in a particular frame.
    # That row is still excluded from evaluation, so occupied rows only need to
    # be a subset of the fixed row definition.
    looks_like_source_rows = True

    hidden_anchor_valid = anchor_valid & hidden_rows_mask
    anchor_valid_pixels = int(anchor_valid.sum())
    anchor_hidden_valid_pixels = int(hidden_anchor_valid.sum())
    anchor_hidden_ratio = (
        anchor_hidden_valid_pixels / anchor_valid_pixels if anchor_valid_pixels else 0.0
    )
    force_anchor_hidden_possible = bool(anchor_hidden_valid_pixels > 0)

    hidden_eval = gt_valid & hidden_rows_mask & pred_valid
    hidden_eval_pixels = int(hidden_eval.sum())
    hidden_gt_valid_pixels = int((gt_valid & hidden_rows_mask).sum())
    if hidden_eval_pixels:
        abs_err = np.abs(pred[hidden_eval].astype(np.float64) - gt[hidden_eval].astype(np.float64))
        zero_ratio = float(np.mean(abs_err <= float(args.leakage_zero_eps)))
        err_5cm_ratio = float(np.mean(abs_err <= 0.05))
        hidden_mae = float(np.mean(abs_err))
        hidden_rmse = float(np.sqrt(np.mean(abs_err * abs_err)))
        hidden_p90 = float(np.percentile(abs_err, 90))
        hidden_p95 = float(np.percentile(abs_err, 95))
        hidden_p99 = float(np.percentile(abs_err, 99))
        sum_abs = float(np.sum(abs_err))
        sum_sq = float(np.sum(abs_err * abs_err))
    else:
        abs_err = np.array([], dtype=np.float64)
        zero_ratio = np.nan
        err_5cm_ratio = np.nan
        hidden_mae = hidden_rmse = hidden_p90 = hidden_p95 = hidden_p99 = np.nan
        sum_abs = sum_sq = 0.0

    warnings = []
    dangers = []
    if not looks_like_source_rows:
        warnings.append("anchor_valid_row_count_mismatch")
    if force_anchor_hidden_possible:
        dangers.append("anchor_valid_in_hidden_rows")
    if np.isfinite(zero_ratio) and zero_ratio > float(args.leakage_warn_err_zero_ratio):
        dangers.append("hidden_zero_error_ratio_high")
    if np.isfinite(err_5cm_ratio) and err_5cm_ratio > float(args.leakage_warn_err_5cm_ratio):
        dangers.append("hidden_5cm_error_ratio_high")

    status = "danger" if dangers else ("warning" if warnings else "ok")
    row = {
        "schema_version": SCHEMA_VERSION,
        "name": f"{scene_id:06d}",
        "status": status,
        "warning": ";".join(warnings),
        "danger": ";".join(dangers),
        "target_rows": int(gt.shape[0]),
        "source_rows": int(args.source_rows),
        "expected_anchor_rows": expected_row_count,
        "anchor_valid_rows": int(len(rows)),
        "anchor_valid_rows_ratio_to_source": (
            int(len(rows)) / int(args.source_rows) if int(args.source_rows) else np.nan
        ),
        "anchor_path_looks_like_source_rows": bool(looks_like_source_rows),
        "anchor_valid_pixels": anchor_valid_pixels,
        "anchor_valid_pixels_in_anchor_rows": int((anchor_valid & anchor_rows_mask).sum()),
        "anchor_valid_pixels_in_hidden_rows": anchor_hidden_valid_pixels,
        "anchor_hidden_valid_ratio": anchor_hidden_ratio,
        "force_anchor_hidden_possible": force_anchor_hidden_possible,
        "hidden_gt_valid_pixels": hidden_gt_valid_pixels,
        "hidden_eval_pixels": hidden_eval_pixels,
        "hidden_mae": hidden_mae,
        "hidden_rmse": hidden_rmse,
        "hidden_p90": hidden_p90,
        "hidden_p95": hidden_p95,
        "hidden_p99": hidden_p99,
        "hidden_err_zero_ratio": zero_ratio,
        "hidden_err_5cm_ratio": err_5cm_ratio,
        "warn_err_zero_ratio": float(args.leakage_warn_err_zero_ratio),
        "warn_err_5cm_ratio": float(args.leakage_warn_err_5cm_ratio),
    }
    accum = {
        "hidden_eval_pixels": hidden_eval_pixels,
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "abs_err": abs_err,
    }
    return row, accum


def metric_row(
    scene_id,
    method,
    area,
    pred,
    gt,
    area_support,
    guide_valid,
    source_valid,
    args,
):
    gt_valid = valid_range_mask(gt, args.range_min, args.range_max)
    pred_valid = valid_range_mask(pred, args.range_min, args.range_max)
    area_gt = gt_valid & area_support
    area_pred = pred_valid & area_support
    eval_mask = area_gt & pred_valid

    errors = pred[eval_mask].astype(np.float64) - gt[eval_mask].astype(np.float64)
    abs_errors = np.abs(errors)
    eval_pixels = int(eval_mask.sum())
    gt_valid_pixels = int(gt_valid.sum())
    area_gt_pixels = int(area_gt.sum())
    area_pred_pixels = int(area_pred.sum())
    missing_prediction_pixels = int((area_gt & ~pred_valid).sum())

    if eval_pixels:
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors * errors)))
        median_abs = float(np.median(abs_errors))
        p90_abs = float(np.percentile(abs_errors, 90))
        p95_abs = float(np.percentile(abs_errors, 95))
        p99_abs = float(np.percentile(abs_errors, 99))
        max_abs = float(np.max(abs_errors))
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        sum_abs = float(np.sum(abs_errors))
        sum_sq = float(np.sum(errors * errors))
    else:
        mae = rmse = median_abs = p90_abs = p95_abs = p99_abs = max_abs = math.nan
        mean_error = std_error = math.nan
        sum_abs = sum_sq = 0.0

    a = source_valid & guide_valid
    b = source_valid & ~guide_valid
    c = ~source_valid & guide_valid
    d = ~source_valid & ~guide_valid
    total_pixels = int(source_valid.size)

    row = {
        "schema_version": SCHEMA_VERSION,
        "name": f"{scene_id:06d}",
        "method": method,
        "area": area,
        "area_legacy": "",
        "mae": mae,
        "rmse": rmse,
        "median_abs": median_abs,
        "p90_abs": p90_abs,
        "p95_abs": p95_abs,
        "p99_abs": p99_abs,
        "max_abs": max_abs,
        "mean_error": mean_error,
        "std_error": std_error,
        "eval_pixels": eval_pixels,
        "gt_valid_pixels": gt_valid_pixels,
        "eval_ratio_over_gt": (eval_pixels / gt_valid_pixels) if gt_valid_pixels else 0.0,
        "area_gt_valid_pixels": area_gt_pixels,
        "area_pred_valid_pixels": area_pred_pixels,
        "area_coverage": (eval_pixels / area_gt_pixels) if area_gt_pixels else 0.0,
        "coverage_in_area": (eval_pixels / area_gt_pixels) if area_gt_pixels else 0.0,
        "missing_prediction_pixels": missing_prediction_pixels,
        "eval_domain": args.eval_domain,
        "sigma_guide": args.sigma_guide,
        "sigma_row": args.sigma_row,
        "sigma_col": args.sigma_col,
        "col_radius": args.col_radius,
        "row_offset": args.row_offset,
        "row_stride": args.row_stride,
        "low_h": args.low_h,
        "target_h": args.target_h,
        "missing_gt_pixels": int((area_support & ~gt_valid).sum()),
        "A_low_valid_guide_valid_count": int(a.sum()),
        "A_low_valid_guide_valid_ratio": (int(a.sum()) / total_pixels) if total_pixels else 0.0,
        "B_low_valid_guide_invalid_count": int(b.sum()),
        "B_low_valid_guide_invalid_ratio": (int(b.sum()) / total_pixels) if total_pixels else 0.0,
        "C_low_invalid_guide_valid_count": int(c.sum()),
        "C_low_invalid_guide_valid_ratio": (int(c.sum()) / total_pixels) if total_pixels else 0.0,
        "D_low_invalid_guide_invalid_count": int(d.sum()),
        "D_low_invalid_guide_invalid_ratio": (int(d.sum()) / total_pixels) if total_pixels else 0.0,
    }
    accum = {
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "eval_pixels": eval_pixels,
        "area_gt_valid_pixels": area_gt_pixels,
        "area_pred_valid_pixels": area_pred_pixels,
        "missing_prediction_pixels": missing_prediction_pixels,
    }
    return row, accum


def distance_metric_row(scene_id, area, bin_min, bin_max, method, pred, gt, area_support, args):
    gt_valid = valid_range_mask(gt, args.range_min, args.range_max)
    pred_valid = valid_range_mask(pred, args.range_min, args.range_max)
    gt_bin = (gt >= float(bin_min)) & (gt < float(bin_max))
    area_gt = gt_valid & area_support & gt_bin
    eval_mask = area_gt & pred_valid

    errors = pred[eval_mask].astype(np.float64) - gt[eval_mask].astype(np.float64)
    metrics = error_metrics(errors)
    area_gt_pixels = int(area_gt.sum())
    eval_pixels = int(eval_mask.sum())
    missing_prediction_pixels = int((area_gt & ~pred_valid).sum())
    row = {
        "schema_version": SCHEMA_VERSION,
        "frame": f"{scene_id:06d}",
        "area": area,
        "distance_bin": distance_bin_label(bin_min, bin_max),
        "bin_min": float(bin_min),
        "bin_max": float(bin_max),
        "method": method,
        **metrics,
        "area_gt_valid_pixels": area_gt_pixels,
        "eval_pixels": eval_pixels,
        "missing_prediction_pixels": missing_prediction_pixels,
        "coverage_in_area": (eval_pixels / area_gt_pixels) if area_gt_pixels else 0.0,
    }
    accum = {
        "area_gt_valid_pixels": area_gt_pixels,
        "eval_pixels": eval_pixels,
        "missing_prediction_pixels": missing_prediction_pixels,
        "errors": errors.astype(np.float32, copy=False),
    }
    return row, accum


def add_distance_rows(scene_id, pred_by_method, gt, supports, args):
    rows = []
    accums = []
    for area in DISTANCE_AREAS:
        if area not in supports:
            continue
        area_support = supports[area]
        for bin_min, bin_max in args.parsed_distance_bins:
            for method, pred in pred_by_method.items():
                row, accum = distance_metric_row(
                    scene_id,
                    area,
                    bin_min,
                    bin_max,
                    method,
                    pred,
                    gt,
                    area_support,
                    args,
                )
                rows.append(row)
                accums.append(accum)
    return rows, accums


def safe_delta(value, baseline):
    if not (np.isfinite(value) and np.isfinite(baseline)):
        return math.nan
    return float(value - baseline)


def safe_relative_improve(value, baseline):
    if not (np.isfinite(value) and np.isfinite(baseline)) or baseline == 0:
        return math.nan
    return float((baseline - value) / baseline)


def resolve_raw_method(summary_rows, requested=None):
    available = sorted({row["method"] for row in summary_rows})
    if requested:
        if requested not in available:
            raise RuntimeError(
                f"Requested raw baseline method {requested!r} was not found. "
                f"Available methods: {available}"
            )
        return requested

    for candidate in ("sdn_raw", "sdn_raw_guide", "raw_sdn"):
        if candidate in available:
            return candidate

    raise RuntimeError(
        "Could not identify the raw SDN baseline for distance-summary deltas. "
        f"Available methods: {available}. Pass --raw_method explicitly."
    )


def build_distance_summary(distance_rows, distance_accums, frame_count, raw_method=None):
    grouped = {}
    for row, accum in zip(distance_rows, distance_accums):
        key = (
            row["area"],
            row["distance_bin"],
            float(row["bin_min"]),
            float(row["bin_max"]),
            row["method"],
        )
        item = grouped.setdefault(
            key,
            {
                "area_gt_valid_pixels": 0,
                "eval_pixels": 0,
                "missing_prediction_pixels": 0,
                "errors": [],
            },
        )
        item["area_gt_valid_pixels"] += int(accum["area_gt_valid_pixels"])
        item["eval_pixels"] += int(accum["eval_pixels"])
        item["missing_prediction_pixels"] += int(accum["missing_prediction_pixels"])
        if accum["errors"].size:
            item["errors"].append(accum["errors"])

    rows = []
    metric_by_group = {}
    for key in sorted(grouped):
        area, label, bin_min, bin_max, method = key
        item = grouped[key]
        if item["errors"]:
            errors = np.concatenate(item["errors"]).astype(np.float64, copy=False)
        else:
            errors = np.array([], dtype=np.float64)
        metrics = error_metrics(errors)
        area_gt_pixels = int(item["area_gt_valid_pixels"])
        eval_pixels = int(item["eval_pixels"])
        row = {
            "schema_version": SCHEMA_VERSION,
            "area": area,
            "distance_bin": label,
            "bin_min": bin_min,
            "bin_max": bin_max,
            "method": method,
            "frames": frame_count,
            "area_gt_valid_pixels": area_gt_pixels,
            "eval_pixels": eval_pixels,
            "missing_prediction_pixels": int(item["missing_prediction_pixels"]),
            "coverage_in_area": (eval_pixels / area_gt_pixels) if area_gt_pixels else 0.0,
            **metrics,
        }
        rows.append(row)
        metric_by_group[(area, label, method)] = row

    raw_method = resolve_raw_method(rows, requested=raw_method)
    print(f"[Distance Summary] raw baseline method: {raw_method}")

    for row in rows:
        baseline = metric_by_group.get((row["area"], row["distance_bin"], raw_method))
        if baseline is None:
            baseline = {}
        row["delta_mae_vs_raw"] = safe_delta(row["mae"], baseline.get("mae", math.nan))
        row["rel_mae_improve_vs_raw"] = safe_relative_improve(
            row["mae"], baseline.get("mae", math.nan)
        )
        row["delta_median_vs_raw"] = safe_delta(
            row["median_abs"], baseline.get("median_abs", math.nan)
        )
        row["rel_median_improve_vs_raw"] = safe_relative_improve(
            row["median_abs"], baseline.get("median_abs", math.nan)
        )
        row["delta_p90_vs_raw"] = safe_delta(row["p90_abs"], baseline.get("p90_abs", math.nan))
        row["rel_p90_improve_vs_raw"] = safe_relative_improve(
            row["p90_abs"], baseline.get("p90_abs", math.nan)
        )
        row["delta_p95_vs_raw"] = safe_delta(row["p95_abs"], baseline.get("p95_abs", math.nan))
        row["rel_p95_improve_vs_raw"] = safe_relative_improve(
            row["p95_abs"], baseline.get("p95_abs", math.nan)
        )
    return rows


def print_distance_summary_table(summary_rows, area="hidden_rows_valid"):
    rows_by_key = {(row["distance_bin"], row["method"]): row for row in summary_rows if row["area"] == area}
    bins = []
    methods = []
    for row in summary_rows:
        if row["area"] != area:
            continue
        if row["distance_bin"] not in bins:
            bins.append(row["distance_bin"])
        if row["method"] not in methods:
            methods.append(row["method"])
    if not bins:
        return
    print(f"\n[Distance Bin Summary] area={area}")
    print("distance_bin | " + " | ".join(f"{method}_mae" for method in methods))
    for label in bins:
        values = [rows_by_key.get((label, method), {}).get("mae", math.nan) for method in methods]
        print(f"{label} | " + " | ".join(str(value) for value in values))


def area_supports(scene_id, gt, guide, source, args):
    gt_valid = valid_range_mask(gt, args.range_min, args.range_max)
    guide_valid = valid_range_mask(guide, args.range_min, args.range_max)
    source_valid = valid_range_mask(source, args.range_min, args.range_max)
    anchor_rows, hidden_rows = row_masks(gt.shape, args, scene_id)
    if int(gt_valid.sum()) > 0:
        hidden_ratio = int((gt_valid & hidden_rows).sum()) / int(gt_valid.sum())
        if hidden_ratio < 0.1:
            print(
                f"WARNING: hidden row area is unexpectedly small: "
                f"{int((gt_valid & hidden_rows).sum())}/{int(gt_valid.sum())}"
            )
    return {
        "full_valid": np.ones_like(gt_valid, dtype=bool),
        "anchor_rows_valid": anchor_rows,
        "hidden_rows_valid": hidden_rows,
    }, guide_valid, source_valid, gt_valid


def save_range_preview(path, range_img, args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import cm
    from matplotlib import image as mpimg

    vmin = float(args.preview_range_min if args.preview_range_min is not None else args.range_min)
    vmax = float(args.preview_range_max if args.preview_range_max is not None else args.range_max)
    if vmax <= vmin:
        raise ValueError("--preview_range_max must be greater than --preview_range_min")

    valid = valid_range_mask(range_img, vmin, vmax)
    norm = np.clip((range_img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.get_cmap(args.preview_cmap)(norm, bytes=True)
    rgba[~valid] = np.array([0, 0, 0, 255], dtype=np.uint8)

    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mpimg.imsave(path, rgba)


def save_previews(scene_id, pred_by_method, args):
    if not args.preview_output_dir:
        return 0
    saved = 0
    methods = args.preview_method or []
    for method in methods:
        if method not in pred_by_method:
            continue
        output_path = osp.join(args.preview_output_dir, method, f"{scene_id:06d}.png")
        save_range_preview(output_path, pred_by_method[method], args)
        saved += 1
    return saved


def write_csv(path, fieldnames, rows):
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(metric_rows, accum_rows, frame_count):
    grouped_rows = defaultdict(list)
    grouped_accum = defaultdict(lambda: defaultdict(float))
    for row, accum in zip(metric_rows, accum_rows):
        key = (row["eval_domain"], row["method"], row["area"])
        grouped_rows[key].append(row)
        for name, value in accum.items():
            grouped_accum[key][name] += value

    summary = []
    for key in sorted(grouped_rows):
        eval_domain, method, area = key
        rows = grouped_rows[key]
        acc = grouped_accum[key]
        eval_pixels = int(acc["eval_pixels"])
        area_gt = int(acc["area_gt_valid_pixels"])
        summary.append(
            {
                "eval_domain": eval_domain,
                "method": method,
                "area": area,
                "schema_version": SCHEMA_VERSION,
                "area_legacy": "",
                "frames": frame_count,
                "frames_total": frame_count,
                "frames_with_eval": sum(1 for row in rows if int(row["eval_pixels"]) > 0),
                "mae_mean": safe_mean([row["mae"] for row in rows]),
                "mae_weighted": (acc["sum_abs"] / eval_pixels) if eval_pixels else math.nan,
                "rmse_mean": safe_mean([row["rmse"] for row in rows]),
                "rmse_weighted": math.sqrt(acc["sum_sq"] / eval_pixels) if eval_pixels else math.nan,
                "median_abs_mean": safe_mean([row["median_abs"] for row in rows]),
                "p90_abs_mean": safe_mean([row["p90_abs"] for row in rows]),
                "p95_abs_mean": safe_mean([row["p95_abs"] for row in rows]),
                "p99_abs_mean": safe_mean([row["p99_abs"] for row in rows]),
                "max_abs_mean": safe_mean([row["max_abs"] for row in rows]),
                "mean_error_mean": safe_mean([row["mean_error"] for row in rows]),
                "std_error_mean": safe_mean([row["std_error"] for row in rows]),
                "eval_pixels": eval_pixels,
                "area_gt_valid_pixels": area_gt,
                "area_pred_valid_pixels": int(acc["area_pred_valid_pixels"]),
                "area_coverage": (eval_pixels / area_gt) if area_gt else 0.0,
                "coverage_in_area": (eval_pixels / area_gt) if area_gt else 0.0,
                "missing_prediction_pixels": int(acc["missing_prediction_pixels"]),
            }
        )
    return summary


def default_leakage_check_path(args):
    if args.leakage_output_csv:
        return args.leakage_output_csv
    directory = osp.dirname(args.summary_csv) or "."
    return osp.join(directory, "range_gdc_leakage_check.csv")


def default_leakage_summary_path(args):
    if args.leakage_summary_csv:
        return args.leakage_summary_csv
    directory = osp.dirname(default_leakage_check_path(args)) or "."
    return osp.join(directory, "range_gdc_leakage_summary.csv")


def default_distance_metrics_path(args):
    if args.distance_metrics_csv:
        return args.distance_metrics_csv
    directory = osp.dirname(args.metrics_csv) or "."
    return osp.join(directory, "guide_r64_distance_metrics.csv")


def default_distance_summary_path(args):
    if args.distance_summary_csv:
        return args.distance_summary_csv
    directory = osp.dirname(default_distance_metrics_path(args)) or "."
    return osp.join(directory, "guide_r64_distance_summary.csv")


def percentile_or_nan(values, percentile):
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def build_leakage_summary(leakage_rows, leakage_accums, leakage_method):
    if not leakage_rows:
        return [
            {
                "schema_version": SCHEMA_VERSION,
                "leakage_method": leakage_method,
                "frames_total": 0,
                "frames_with_warning": 0,
                "frames_with_danger": 0,
                "mean_num_anchor_rows": math.nan,
                "max_num_anchor_rows": math.nan,
                "sum_anchor_valid_in_hidden_rows": 0,
                "mean_hidden_err_lt_1e-6_ratio": math.nan,
                "mean_hidden_err_lt_5e-2_ratio": math.nan,
                "weighted_hidden_mae": math.nan,
                "weighted_hidden_rmse": math.nan,
                "weighted_hidden_p90": math.nan,
                "weighted_hidden_p95": math.nan,
                "weighted_hidden_p99": math.nan,
                "status": "skipped",
            }
        ]

    total_pixels = sum(int(acc["hidden_eval_pixels"]) for acc in leakage_accums)
    total_abs = sum(float(acc["sum_abs"]) for acc in leakage_accums)
    total_sq = sum(float(acc["sum_sq"]) for acc in leakage_accums)
    if total_pixels:
        hidden_errors = np.concatenate(
            [acc["abs_err"] for acc in leakage_accums if acc["abs_err"].size > 0]
        )
    else:
        hidden_errors = np.array([], dtype=np.float64)
    danger_count = sum(1 for row in leakage_rows if row["status"] == "danger")
    warning_count = sum(1 for row in leakage_rows if row["status"] in {"warning", "danger"})
    status = "DANGER" if danger_count else ("WARNING" if warning_count else "OK")
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "leakage_method": leakage_method,
            "frames_total": len(leakage_rows),
            "frames_with_warning": warning_count,
            "frames_with_danger": danger_count,
            "mean_num_anchor_rows": safe_mean([row["anchor_valid_rows"] for row in leakage_rows]),
            "max_num_anchor_rows": max(int(row["anchor_valid_rows"]) for row in leakage_rows),
            "sum_anchor_valid_in_hidden_rows": sum(
                int(row["anchor_valid_pixels_in_hidden_rows"]) for row in leakage_rows
            ),
            "mean_hidden_err_lt_1e-6_ratio": safe_mean(
                [row["hidden_err_zero_ratio"] for row in leakage_rows]
            ),
            "mean_hidden_err_lt_5e-2_ratio": safe_mean(
                [row["hidden_err_5cm_ratio"] for row in leakage_rows]
            ),
            "weighted_hidden_mae": (total_abs / total_pixels) if total_pixels else math.nan,
            "weighted_hidden_rmse": math.sqrt(total_sq / total_pixels) if total_pixels else math.nan,
            "weighted_hidden_p90": percentile_or_nan(hidden_errors, 90),
            "weighted_hidden_p95": percentile_or_nan(hidden_errors, 95),
            "weighted_hidden_p99": percentile_or_nan(hidden_errors, 99),
            "status": status,
        }
    ]


def print_leakage_summary(summary_row, leakage_output_csv):
    print("\n[Leakage Check]")
    for key in (
        "frames_total",
        "frames_with_warning",
        "frames_with_danger",
        "mean_num_anchor_rows",
        "sum_anchor_valid_in_hidden_rows",
        "mean_hidden_err_lt_1e-6_ratio",
        "mean_hidden_err_lt_5e-2_ratio",
    ):
        print(f"{key}: {summary_row[key]}")
    print(f"status: {summary_row['status']}")
    if summary_row["status"] == "DANGER":
        print("\n[Leakage Check][DANGER]")
        print("Possible R64/full GT leakage detected.")
        print(f"See: {leakage_output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--gt_range_path", required=True)
    parser.add_argument("--guide_range_path", required=True)
    parser.add_argument("--source_range_path", default=None)
    parser.add_argument("--method", action="append", required=True, help="name=range_npy_dir")
    parser.add_argument(
        "--raw_method",
        default=None,
        help=(
            "Method name used as the raw SDN baseline for distance-summary "
            "delta/improvement columns. If omitted, the evaluator auto-detects "
            "sdn_raw, sdn_raw_guide, or raw_sdn."
        ),
    )
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--eval_domain", default="guide64_and_range64_valid")
    parser.add_argument("--range_min", type=float, default=0.1)
    parser.add_argument("--range_max", type=float, default=80.0)
    parser.add_argument("--sigma_guide", default="")
    parser.add_argument("--sigma_row", default="")
    parser.add_argument("--sigma_col", default="")
    parser.add_argument("--col_radius", default=1)
    parser.add_argument("--row_offset", type=int, default=None)
    parser.add_argument("--row_stride", type=int, default=None)
    parser.add_argument("--source_row_indices", type=int, nargs="+", default=None)
    parser.add_argument("--anchor_rows", dest="source_row_indices", type=int, nargs="+")
    parser.add_argument("--selected_rows_dir", default=None)
    parser.add_argument("--projection_meta_path", default=None)
    parser.add_argument("--source_rows", type=int, default=32)
    parser.add_argument("--low_h", default=4)
    parser.add_argument("--target_h", type=int, default=64)
    parser.add_argument("--expected_height", type=int, default=64)
    parser.add_argument("--expected_width", type=int, default=1024)
    parser.add_argument("--expected_frame_count", type=int, default=None)
    parser.add_argument("--enable_leakage_check", action="store_true")
    parser.add_argument("--leakage_output_csv", default=None)
    parser.add_argument("--leakage_summary_csv", default=None)
    parser.add_argument("--anchor_range_path", default=None)
    parser.add_argument("--leakage_method", default="sdn_range_gdc")
    parser.add_argument("--leakage_warn_err_zero_ratio", type=float, default=0.01)
    parser.add_argument("--leakage_warn_err_5cm_ratio", type=float, default=0.8)
    parser.add_argument("--leakage_zero_eps", type=float, default=1e-6)
    parser.add_argument("--leakage_anchor_row_tolerance", type=int, default=1)
    parser.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Write diagnostics but do not fail when leakage checks report danger.",
    )
    parser.add_argument("--preview_output_dir", default=None)
    parser.add_argument("--preview_method", action="append", default=[])
    parser.add_argument("--preview_max_items", type=int, default=None)
    parser.add_argument("--preview_range_min", type=float, default=None)
    parser.add_argument("--preview_range_max", type=float, default=None)
    parser.add_argument("--preview_cmap", default="turbo")
    parser.add_argument("--target_rows", dest="target_h", type=int)
    parser.add_argument("--enable_distance_bins", action="store_true")
    parser.add_argument("--distance_bins", default="0,10,20,30,40,50,60,70,80")
    parser.add_argument("--distance_metrics_csv", default=None)
    parser.add_argument("--distance_summary_csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    projection_meta = load_projection_meta(args.projection_meta_path)
    args.projection_selected_rows = (
        projection_meta["selected_rows"].astype(np.int32)
        if projection_meta is not None and "selected_rows" in projection_meta.files
        else None
    )
    methods = [parse_method(value) for value in args.method]
    source_path = args.source_range_path or args.guide_range_path

    metric_rows = []
    accum_rows = []
    distance_rows = []
    distance_accums = []
    leakage_rows = []
    leakage_accums = []
    common_eval_counts = defaultdict(dict)
    scene_ids = read_split_ids(args.split_file)
    if not scene_ids:
        raise ValueError("split file contains no frames")
    expected_frames = len(scene_ids) if args.expected_frame_count is None else int(args.expected_frame_count)
    if len(scene_ids) != expected_frames:
        raise ValueError(
            f"Evaluation frame count mismatch: split has {len(scene_ids)}, expected {expected_frames}"
        )
    leakage_active = bool(args.enable_leakage_check)
    if leakage_active and not args.anchor_range_path:
        print(
            "WARNING: --enable_leakage_check was set but --anchor_range_path is missing; "
            "skipping leakage diagnostics."
        )
        leakage_active = False
    leakage_output_csv = default_leakage_check_path(args)
    leakage_summary_csv = default_leakage_summary_path(args)
    distance_active = bool(args.enable_distance_bins)
    if distance_active:
        args.parsed_distance_bins = parse_distance_bins(args.distance_bins)
    distance_metrics_csv = default_distance_metrics_path(args)
    distance_summary_csv = default_distance_summary_path(args)
    preview_frames_saved = 0
    preview_images_saved = 0
    preview_available = True
    for scene_id in scene_ids:
        print(f"Evaluating scene {scene_id:06d}...")
        gt = np.load(find_input_npy(args.gt_range_path, scene_id)).astype(np.float32)
        guide = np.load(find_input_npy(args.guide_range_path, scene_id)).astype(np.float32)
        source = np.load(find_input_npy(source_path, scene_id)).astype(np.float32)
        anchor = None
        if leakage_active:
            anchor = np.load(find_input_npy(args.anchor_range_path, scene_id)).astype(np.float32)
            if anchor.shape != gt.shape:
                raise ValueError(
                    f"{scene_id:06d}: anchor shape {anchor.shape} != gt shape {gt.shape}"
                )
        supports, guide_valid, source_valid, gt_valid = area_supports(scene_id, gt, guide, source, args)
        pred_by_method = {}
        valid_by_method = {}
        for method, path in methods:
            pred_by_method[method] = np.load(find_input_npy(path, scene_id)).astype(np.float32)
            pred = pred_by_method[method]
            if pred.shape != gt.shape:
                raise ValueError(
                    f"{scene_id:06d} {method}: pred shape {pred.shape} != gt shape {gt.shape}"
                )
            valid_by_method[method] = valid_range_mask(pred, args.range_min, args.range_max)
        if scene_id == scene_ids[0]:
            validate_matching_projection_metadata(
                scene_id,
                [("gt", args.gt_range_path), ("anchor", args.guide_range_path)] + methods,
            )
            validate_projection_meta_once(projection_meta, gt, guide, pred_by_method, args)
        all_pred_valid = np.ones_like(gt_valid, dtype=bool)
        for mask in valid_by_method.values():
            all_pred_valid &= mask
        supports["common_full_valid"] = gt_valid & all_pred_valid
        supports["common_hidden_valid"] = supports["hidden_rows_valid"] & gt_valid & all_pred_valid

        preview_allowed = (
            args.preview_output_dir
            and preview_available
            and (
                args.preview_max_items is None
                or preview_frames_saved < int(args.preview_max_items)
            )
        )
        if preview_allowed:
            try:
                saved = save_previews(scene_id, pred_by_method, args)
            except ImportError as exc:
                print(f"WARNING: matplotlib is unavailable; skipping range previews: {exc}")
                preview_available = False
                saved = 0
            if saved:
                preview_images_saved += saved
                preview_frames_saved += 1

        for method, _ in methods:
            pred = pred_by_method[method]
            for area in AREAS:
                row, accum = metric_row(
                    scene_id,
                    method,
                    area,
                    pred,
                    gt,
                    supports[area],
                    guide_valid,
                    source_valid,
                    args,
                )
                if area.startswith("common"):
                    common_eval_counts[(scene_id, area)][method] = int(row["eval_pixels"])
                metric_rows.append(row)
                accum_rows.append(accum)
        if distance_active:
            rows, accums = add_distance_rows(scene_id, pred_by_method, gt, supports, args)
            distance_rows.extend(rows)
            distance_accums.extend(accums)
        if leakage_active:
            leak, leak_accum = leakage_row(scene_id, pred_by_method, gt, anchor, args)
        else:
            leak, leak_accum = None, None
        if leak is not None:
            leakage_rows.append(leak)
            leakage_accums.append(leak_accum)

    for (scene_id, area), counts in common_eval_counts.items():
        if len(set(counts.values())) > 1:
            raise RuntimeError(
                f"Common mask eval_pixels mismatch for {scene_id:06d} {area}: {counts}"
            )

    summary_rows = build_summary(metric_rows, accum_rows, len(scene_ids))
    write_csv(args.metrics_csv, METRICS_FIELDNAMES, metric_rows)
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    if distance_active:
        distance_summary_rows = build_distance_summary(
            distance_rows,
            distance_accums,
            len(scene_ids),
            raw_method=args.raw_method,
        )
        write_csv(distance_metrics_csv, DISTANCE_METRICS_FIELDNAMES, distance_rows)
        write_csv(distance_summary_csv, DISTANCE_SUMMARY_FIELDNAMES, distance_summary_rows)
        print(f"Saved distance metrics: {distance_metrics_csv}")
        print(f"Saved distance summary: {distance_summary_csv}")
        print_distance_summary_table(distance_summary_rows, area="hidden_rows_valid")
        print_distance_summary_table(distance_summary_rows, area="common_hidden_valid")
    if leakage_active:
        leakage_summary_rows = build_leakage_summary(
            leakage_rows,
            leakage_accums,
            args.leakage_method,
        )
        write_csv(leakage_output_csv, LEAKAGE_FIELDNAMES, leakage_rows)
        write_csv(leakage_summary_csv, LEAKAGE_SUMMARY_FIELDNAMES, leakage_summary_rows)
        print_leakage_summary(leakage_summary_rows[0], leakage_output_csv)
        print(f"Saved leakage diagnostics: {leakage_output_csv}")
        print(f"Saved leakage summary: {leakage_summary_csv}")
        if leakage_summary_rows[0]["status"] == "DANGER" and not args.allow_leakage:
            raise SystemExit(
                "Leakage validation failed. Use --allow_leakage only for an explicitly "
                "documented diagnostic override."
            )
    print(f"Saved metrics: {args.metrics_csv}")
    print(f"Saved summary: {args.summary_csv}")
    if args.preview_output_dir and preview_images_saved:
        print(
            f"Saved range previews: {preview_images_saved} image(s) "
            f"for {preview_frames_saved} frame(s) under {args.preview_output_dir}"
        )


if __name__ == "__main__":
    main()
