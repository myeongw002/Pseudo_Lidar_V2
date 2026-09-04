#!/usr/bin/env python3
"""Oracle diagnostic for understanding when RGC corrections help or harm.

This script only reads existing range-grid artifacts.  Ground truth is used to
measure an unattainable per-cell Raw/RGC selection upper bound and to classify
harm as wrong-direction correction, overshoot, or correction of an already
accurate Raw prediction.  Oracle quantities are diagnostics, not inference
results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGIONS = ("overall", "boundary", "interior")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    edges: tuple[float, ...]
    unit: str = ""


FEATURES = (
    FeatureSpec("raw_abs_error", (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, math.inf), "m"),
    FeatureSpec("abs_correction", (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, math.inf), "m"),
    FeatureSpec("gt_range", (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.000001), "m"),
    FeatureSpec("predicted_log_contrast", (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, math.inf), ""),
    FeatureSpec("valid_neighbor_count", (-0.5, 1.5, 3.5, 5.5, 7.5, 8.5), "neighbors"),
    FeatureSpec("accepted_anchor_distance", (0.0, 1.5, 2.5, 3.5, 4.5, 8.5, math.inf), "grid cells"),
)


def valid_range_mask(array: np.ndarray, range_min: float,
                     range_max: float) -> np.ndarray:
    return (
        np.isfinite(array)
        & (array >= float(range_min))
        & (array <= float(range_max))
    )


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


def read_split_scene_ids(split_file: str | Path) -> list[str]:
    with Path(split_file).open() as handle:
        scene_ids = [f"{int(line.strip()):06d}" for line in handle if line.strip()]
    if not scene_ids:
        raise ValueError(f"No scene ids in split file: {split_file}")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"Duplicate scene ids in split file: {split_file}")
    return sorted(scene_ids)


def hidden_rows_mask(shape: tuple[int, int], source_rows: list[int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    rows = np.asarray(source_rows, dtype=np.int64)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= shape[0]):
        raise ValueError(f"source_rows must be within [0, {shape[0]})")
    mask[rows, :] = False
    return mask


def gt_boundary_mask(gt: np.ndarray, range_min: float, range_max: float,
                     boundary_log_thr: float) -> np.ndarray:
    valid = valid_range_mask(gt, range_min, range_max)
    log_gt = np.zeros(gt.shape, dtype=np.float64)
    log_gt[valid] = np.log(gt[valid])
    boundary = np.zeros(gt.shape, dtype=bool)

    right_pair = (
        valid
        & np.roll(valid, -1, axis=1)
        & (np.abs(log_gt - np.roll(log_gt, -1, axis=1)) > boundary_log_thr)
    )
    boundary |= right_pair
    boundary |= np.roll(right_pair, 1, axis=1)
    if gt.shape[0] > 1:
        vertical_pair = (
            valid[:-1]
            & valid[1:]
            & (np.abs(log_gt[:-1] - log_gt[1:]) > boundary_log_thr)
        )
        boundary[:-1] |= vertical_pair
        boundary[1:] |= vertical_pair
    return boundary


def periodic_boundary_distance(boundary: np.ndarray) -> np.ndarray:
    return periodic_distance_to_mask(boundary)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def format_edge(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}"


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


def periodic_distance_to_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float64)
    width = mask.shape[1]
    tiled = np.tile(mask, (1, 3))
    return distance_transform_edt(~tiled)[:, width:2 * width]


def local_predicted_features(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum local log-range contrast and valid 8-neighbor count."""
    log_raw = np.zeros(raw.shape, dtype=np.float64)
    log_raw[valid] = np.log(raw[valid])
    contrast = np.zeros(raw.shape, dtype=np.float64)
    degree = np.zeros(raw.shape, dtype=np.int16)
    height, _ = raw.shape

    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        horizontal_valid = np.roll(valid, -dc, axis=1)
        horizontal_log = np.roll(log_raw, -dc, axis=1)
        shifted_valid = np.zeros(valid.shape, dtype=bool)
        shifted_log = np.zeros(log_raw.shape, dtype=np.float64)
        if dr < 0:
            shifted_valid[-dr:, :] = horizontal_valid[:dr, :]
            shifted_log[-dr:, :] = horizontal_log[:dr, :]
        elif dr > 0:
            shifted_valid[:-dr, :] = horizontal_valid[dr:, :]
            shifted_log[:-dr, :] = horizontal_log[dr:, :]
        else:
            shifted_valid = horizontal_valid
            shifted_log = horizontal_log
        pair = valid & shifted_valid
        degree[pair] += 1
        diff = np.abs(log_raw - shifted_log)
        contrast[pair] = np.maximum(contrast[pair], diff[pair])

    contrast[~valid] = np.nan
    return contrast, degree.astype(np.float64)


def classify_harm_modes(
    delta: np.ndarray,
    target: np.ndarray,
    harmed: np.ndarray,
    target_zero_tol: float,
) -> dict[str, np.ndarray]:
    """Partition harmed cells into mutually exclusive oracle failure modes."""
    near_zero = harmed & (np.abs(target) <= target_zero_tol)
    remaining = harmed & ~near_zero
    wrong_direction = remaining & (delta * target < 0.0)
    overshoot = (
        remaining
        & (delta * target > 0.0)
        & (np.abs(delta) > 2.0 * np.abs(target))
    )
    other = harmed & ~(near_zero | wrong_direction | overshoot)
    return {
        "harmed_near_zero_target": near_zero,
        "harmed_wrong_direction": wrong_direction,
        "harmed_overshoot": overshoot,
        "harmed_other_numeric": other,
    }


def update_metric(acc: dict[str, float], raw_abs: np.ndarray,
                  rgc_abs: np.ndarray, correction: np.ndarray,
                  correction_tol: float, error_tol: float) -> None:
    error_change = rgc_abs - raw_abs
    improved = error_change < -error_tol
    harmed = error_change > error_tol
    neutral = ~(improved | harmed)
    active = np.abs(correction) > correction_tol

    acc["pixels"] += int(raw_abs.size)
    acc["raw_abs_sum"] += float(np.sum(raw_abs))
    acc["rgc_abs_sum"] += float(np.sum(rgc_abs))
    acc["oracle_abs_sum"] += float(np.sum(np.minimum(raw_abs, rgc_abs)))
    acc["active_pixels"] += int(np.sum(active))
    acc["improved_pixels"] += int(np.sum(improved))
    acc["harmed_pixels"] += int(np.sum(harmed))
    acc["neutral_pixels"] += int(np.sum(neutral))
    acc["total_improvement"] += float(np.sum(np.maximum(-error_change, 0.0)))
    acc["total_harm"] += float(np.sum(np.maximum(error_change, 0.0)))
    acc["abs_correction_sum"] += float(np.sum(np.abs(correction)))


def metric_row(prefix: dict, acc: dict[str, float]) -> dict:
    pixels = int(acc["pixels"])
    raw_sum = acc["raw_abs_sum"]
    rgc_sum = acc["rgc_abs_sum"]
    oracle_sum = acc["oracle_abs_sum"]
    raw_gain = raw_sum - rgc_sum
    oracle_gain = raw_sum - oracle_sum
    headroom = rgc_sum - oracle_sum
    return {
        **prefix,
        "pixels": pixels,
        "raw_mae": safe_ratio(raw_sum, pixels),
        "rgc_mae": safe_ratio(rgc_sum, pixels),
        "oracle_mae": safe_ratio(oracle_sum, pixels),
        "rgc_gain_vs_raw": safe_ratio(raw_gain, pixels),
        "oracle_gain_vs_raw": safe_ratio(oracle_gain, pixels),
        "recoverable_harm_per_pixel": safe_ratio(headroom, pixels),
        "captured_oracle_gain_ratio": safe_ratio(raw_gain, oracle_gain),
        "active_pixels": int(acc["active_pixels"]),
        "active_ratio": safe_ratio(acc["active_pixels"], pixels),
        "improved_pixels": int(acc["improved_pixels"]),
        "improved_ratio": safe_ratio(acc["improved_pixels"], pixels),
        "harmed_pixels": int(acc["harmed_pixels"]),
        "harm_rate": safe_ratio(acc["harmed_pixels"], pixels),
        "neutral_pixels": int(acc["neutral_pixels"]),
        "total_improvement": acc["total_improvement"],
        "total_harm": acc["total_harm"],
        "net_error_reduction": acc["total_improvement"] - acc["total_harm"],
        "improvement_to_harm_ratio": safe_ratio(
            acc["total_improvement"], acc["total_harm"]
        ),
        "mean_abs_correction": safe_ratio(acc["abs_correction_sum"], pixels),
    }


def update_mode(acc: dict[str, float], selected: np.ndarray,
                error_change: np.ndarray, correction: np.ndarray,
                raw_abs: np.ndarray) -> None:
    count = int(np.sum(selected))
    acc["pixels"] += count
    if not count:
        return
    acc["total_harm"] += float(np.sum(error_change[selected]))
    acc["raw_abs_sum"] += float(np.sum(raw_abs[selected]))
    acc["abs_correction_sum"] += float(np.sum(np.abs(correction[selected])))


def mode_rows(accumulators: dict, region_harmed: dict[str, int],
              region_harm_sum: dict[str, float]) -> list[dict]:
    rows = []
    for (region, mode), acc in sorted(accumulators.items()):
        count = int(acc["pixels"])
        rows.append({
            "region": region,
            "failure_mode": mode,
            "pixels": count,
            "share_of_harmed_pixels": safe_ratio(count, region_harmed[region]),
            "total_harm": acc["total_harm"],
            "share_of_total_harm": safe_ratio(acc["total_harm"], region_harm_sum[region]),
            "mean_harm": safe_ratio(acc["total_harm"], count),
            "mean_raw_abs_error": safe_ratio(acc["raw_abs_sum"], count),
            "mean_abs_correction": safe_ratio(acc["abs_correction_sum"], count),
        })
    return rows


def validate_inputs(args: argparse.Namespace, scene_ids: list[str]) -> dict[str, dict[str, str]]:
    directories = {
        "raw": args.raw_range_dir,
        "gt": args.gt_range_dir,
        "rgc": args.rgc_range_dir,
    }
    if args.anchor_range_dir:
        directories["anchor"] = args.anchor_range_dir
    for index, path in enumerate(args.support_range_dir):
        directories[f"support_{index}"] = path

    mappings: dict[str, dict[str, str]] = {}
    for name, directory in directories.items():
        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(path)
        mapping = npy_map(str(path))
        missing = [scene_id for scene_id in scene_ids if scene_id not in mapping]
        if missing:
            raise FileNotFoundError(
                f"{name} missing {len(missing)} split frames; sample={missing[:5]}"
            )
        mappings[name] = mapping
    return mappings


def region_masks(common: np.ndarray, boundary_distance: np.ndarray,
                 boundary_radius: int) -> dict[str, np.ndarray]:
    return {
        "overall": common,
        "boundary": common & (boundary_distance <= boundary_radius),
        "interior": common & (boundary_distance >= 4.0),
    }


def add_condition_bins(accumulators: dict, region: str, feature: FeatureSpec,
                       values: np.ndarray, region_mask: np.ndarray,
                       raw_abs: np.ndarray, rgc_abs: np.ndarray,
                       correction: np.ndarray, correction_tol: float,
                       error_tol: float) -> None:
    finite = np.isfinite(values)
    for index, (low, high) in enumerate(zip(feature.edges[:-1], feature.edges[1:])):
        selected = region_mask & finite & (values >= low) & (values < high)
        key = (region, feature.name, index, low, high, feature.unit)
        vectors = (raw_abs[selected], rgc_abs[selected], correction[selected])
        update_metric(accumulators[key], *vectors, correction_tol, error_tol)


def analyze(args: argparse.Namespace) -> None:
    scene_ids = read_split_scene_ids(args.split_file)
    if args.max_items is not None:
        scene_ids = scene_ids[:args.max_items]
    mappings = validate_inputs(args, scene_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = (
        "oracle_per_frame.csv", "oracle_summary.csv", "harm_mode_summary.csv",
        "condition_summary.csv", "oracle_analysis_metadata.json",
    )
    existing = [name for name in output_files if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}; pass --overwrite to replace them"
        )

    aggregate = defaultdict(lambda: defaultdict(float))
    condition_acc = defaultdict(lambda: defaultdict(float))
    harm_mode_acc = defaultdict(lambda: defaultdict(float))
    per_frame_rows: list[dict] = []
    region_harmed = defaultdict(int)
    region_harm_sum = defaultdict(float)
    source_rows = [int(row) for row in args.source_rows]

    for scene_id in scene_ids:
        arrays = {
            name: np.load(mapping[scene_id]).astype(np.float64)
            for name, mapping in mappings.items()
        }
        shape = arrays["gt"].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
            raise ValueError(f"{scene_id}: all inputs must be 2D and share one shape")

        raw_valid = valid_range_mask(arrays["raw"], args.range_min, args.range_max)
        common = (
            valid_range_mask(arrays["gt"], args.range_min, args.range_max)
            & raw_valid
            & valid_range_mask(arrays["rgc"], args.range_min, args.range_max)
            & hidden_rows_mask(shape, source_rows)
        )
        for name in mappings:
            if name.startswith("support_"):
                common &= valid_range_mask(arrays[name], args.range_min, args.range_max)

        boundary = gt_boundary_mask(
            arrays["gt"], args.range_min, args.range_max, args.boundary_log_thr
        )
        boundary_distance = periodic_boundary_distance(boundary)
        masks = region_masks(common, boundary_distance, args.boundary_radius)

        target = arrays["gt"] - arrays["raw"]
        correction = arrays["rgc"] - arrays["raw"]
        raw_abs = np.abs(target)
        rgc_abs = np.abs(correction - target)
        predicted_contrast, degree = local_predicted_features(arrays["raw"], raw_valid)

        if "anchor" in arrays:
            anchor_valid = valid_range_mask(
                arrays["anchor"], args.range_min, args.range_max
            )
            accepted = (
                anchor_valid
                & raw_valid
                & (np.abs(arrays["anchor"] - arrays["raw"]) < args.anchor_reject_thr)
            )
            anchor_distance = periodic_distance_to_mask(accepted)
        else:
            anchor_distance = np.full(shape, np.nan, dtype=np.float64)

        feature_values = {
            "raw_abs_error": raw_abs,
            "abs_correction": np.abs(correction),
            "gt_range": arrays["gt"],
            "predicted_log_contrast": predicted_contrast,
            "valid_neighbor_count": degree,
            "accepted_anchor_distance": anchor_distance,
        }

        error_change = rgc_abs - raw_abs
        harmed = error_change > args.error_tol
        modes = classify_harm_modes(
            correction, target, harmed, args.target_zero_tol
        )

        for region, mask in masks.items():
            frame_acc = defaultdict(float)
            update_metric(
                frame_acc, raw_abs[mask], rgc_abs[mask], correction[mask],
                args.correction_tol, args.error_tol,
            )
            per_frame_rows.append(metric_row({"frame_id": scene_id, "region": region}, frame_acc))
            update_metric(
                aggregate[region], raw_abs[mask], rgc_abs[mask], correction[mask],
                args.correction_tol, args.error_tol,
            )

            harmed_region = mask & harmed
            region_harmed[region] += int(np.sum(harmed_region))
            region_harm_sum[region] += float(np.sum(error_change[harmed_region]))
            for mode, mode_mask in modes.items():
                update_mode(
                    harm_mode_acc[(region, mode)], mask & mode_mask,
                    error_change, correction, raw_abs,
                )

            for feature in FEATURES:
                if feature.name == "accepted_anchor_distance" and "anchor" not in arrays:
                    continue
                add_condition_bins(
                    condition_acc, region, feature, feature_values[feature.name],
                    mask, raw_abs, rgc_abs, correction,
                    args.correction_tol, args.error_tol,
                )

    summary_rows = [
        metric_row({"region": region, "frames": len(scene_ids)}, aggregate[region])
        for region in REGIONS
    ]
    condition_rows = []
    for (region, feature, index, low, high, unit), acc in sorted(condition_acc.items()):
        condition_rows.append(metric_row({
            "region": region,
            "feature": feature,
            "bin_index": index,
            "bin_low_inclusive": format_edge(low),
            "bin_high_exclusive": format_edge(high),
            "bin_label": f"[{format_edge(low)}, {format_edge(high)})",
            "unit": unit,
        }, acc))

    write_csv(output_dir / "oracle_per_frame.csv", per_frame_rows)
    write_csv(output_dir / "oracle_summary.csv", summary_rows)
    write_csv(
        output_dir / "harm_mode_summary.csv",
        mode_rows(harm_mode_acc, region_harmed, region_harm_sum),
    )
    write_csv(output_dir / "condition_summary.csv", condition_rows)

    metadata = {
        "analysis_kind": "gt_assisted_oracle_diagnostic_not_inference_performance",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split_file": str(Path(args.split_file).resolve()),
        "frames": len(scene_ids),
        "source_rows": source_rows,
        "range_min": args.range_min,
        "range_max": args.range_max,
        "boundary_log_thr": args.boundary_log_thr,
        "boundary_radius": args.boundary_radius,
        "correction_tol": args.correction_tol,
        "error_tol": args.error_tol,
        "target_zero_tol": args.target_zero_tol,
        "anchor_reject_thr": args.anchor_reject_thr,
        "inputs": {
            "raw": str(Path(args.raw_range_dir).resolve()),
            "gt": str(Path(args.gt_range_dir).resolve()),
            "rgc": str(Path(args.rgc_range_dir).resolve()),
            "anchor": str(Path(args.anchor_range_dir).resolve()) if args.anchor_range_dir else None,
            "extra_common_support": [str(Path(path).resolve()) for path in args.support_range_dir],
        },
        "oracle_warning": "Oracle MAE uses GT to select Raw or RGC per cell and is not achievable inference performance.",
        "harm_mode_identity": "For nonzero target residual, a harmful correction is either wrong-direction or same-direction overshoot beyond twice the oracle residual; target-near-zero is reported separately.",
    }
    with (output_dir / "oracle_analysis_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Analyzed {len(scene_ids)} frame(s)")
    for row in summary_rows:
        print(
            f"{row['region']:8s} raw={row['raw_mae']:.6f} "
            f"rgc={row['rgc_mae']:.6f} oracle={row['oracle_mae']:.6f} "
            f"harm={100.0 * row['harm_rate']:.3f}% "
            f"captured={100.0 * row['captured_oracle_gain_ratio']:.2f}%"
        )
    print(f"Saved: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-range-dir", required=True)
    parser.add_argument("--gt-range-dir", required=True)
    parser.add_argument("--rgc-range-dir", required=True)
    parser.add_argument("--anchor-range-dir")
    parser.add_argument(
        "--support-range-dir", action="append", default=[],
        help="Optional additional prediction directory included only in common support; repeatable.",
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rows", type=int, nargs="+", default=[5, 7, 9, 11])
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--boundary-log-thr", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--correction-tol", type=float, default=1e-3)
    parser.add_argument("--error-tol", type=float, default=1e-6)
    parser.add_argument("--target-zero-tol", type=float, default=1e-3)
    parser.add_argument("--anchor-reject-thr", type=float, default=2.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.range_max <= args.range_min:
        parser.error("--range-max must be greater than --range-min")
    for name in ("boundary_log_thr", "correction_tol", "error_tol",
                 "target_zero_tol", "anchor_reject_thr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.boundary_radius < 0:
        parser.error("--boundary-radius must be nonnegative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    if len(set(args.source_rows)) != len(args.source_rows):
        parser.error("--source-rows contains duplicates")
    return args


if __name__ == "__main__":
    analyze(parse_args())
