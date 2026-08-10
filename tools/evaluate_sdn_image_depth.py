#!/usr/bin/env python3
"""Paper-style SDN image-depth sanity evaluation.

This evaluator compares SDN camera-axis depth Z against sparse 64-beam LiDAR
camera-axis depth projected onto the left image. Pixels occupied by the sparse
4-beam landmark depth map are excluded, matching the protocol described for
Pseudo-LiDAR++ Tables 11 and 12 as closely as possible with the current data.

The script reports pooled pixel statistics for each GT-depth interval and also
writes per-frame statistics for basic stability checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PAPER_SDN_MEDIAN = {
    (0.0, 10.0): 0.07,
    (10.0, 20.0): 0.12,
    (20.0, 30.0): 0.30,
    (30.0, 40.0): 0.60,
    (40.0, 50.0): 0.89,
    (50.0, 60.0): 1.31,
    (60.0, 70.0): 1.73,
}

PAPER_SDN_MEAN = {
    (0.0, 10.0): 0.21,
    (10.0, 20.0): 0.35,
    (20.0, 30.0): 0.87,
    (30.0, 40.0): 1.80,
    (40.0, 50.0): 2.67,
    (50.0, 60.0): 4.27,
    (60.0, 70.0): 5.82,
}

PAPER_SDN_STD = {
    (0.0, 10.0): 0.89,
    (10.0, 20.0): 1.16,
    (20.0, 30.0): 2.31,
    (30.0, 40.0): 4.22,
    (40.0, 50.0): 6.00,
    (50.0, 60.0): 8.78,
    (60.0, 70.0): 11.23,
}


@dataclass(frozen=True)
class BinSummary:
    depth_min: float
    depth_max: float
    count: int
    contributing_frames: int
    median_abs_error: float
    mean_abs_error: float
    std_abs_error: float
    rmse: float
    mean_signed_error: float
    p90_abs_error: float
    p95_abs_error: float
    paper_sdn_median: float
    paper_sdn_mean: float
    paper_sdn_std: float
    median_ratio_to_paper: float
    mean_ratio_to_paper: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SDN image-plane camera-axis depth Z against projected "
            "64-beam LiDAR depth while excluding sparse 4-beam landmark pixels."
        )
    )
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--pred_depth_path", required=True, type=Path)
    parser.add_argument("--gt_depth_path", required=True, type=Path)
    parser.add_argument("--landmark_depth_path", required=True, type=Path)
    parser.add_argument("--summary_csv", required=True, type=Path)
    parser.add_argument("--frame_csv", required=True, type=Path)
    parser.add_argument("--metadata_json", type=Path, default=None)
    parser.add_argument(
        "--depth_bins",
        default="0,10,20,30,40,50,60,70,80",
        help="Comma-separated GT camera-depth bin edges in meters.",
    )
    parser.add_argument(
        "--pred_min",
        type=float,
        default=0.0,
        help="Minimum valid predicted depth; default accepts any positive depth.",
    )
    parser.add_argument(
        "--pred_max",
        type=float,
        default=80.0,
        help="Maximum valid predicted depth.",
    )
    parser.add_argument(
        "--gt_min",
        type=float,
        default=0.0,
        help="Minimum valid GT depth.",
    )
    parser.add_argument(
        "--gt_max",
        type=float,
        default=80.0,
        help="Maximum valid GT depth.",
    )
    parser.add_argument(
        "--include_landmarks",
        action="store_true",
        help="Do not exclude sparse landmark pixels. Use only for debugging.",
    )
    parser.add_argument(
        "--strict_shape",
        action="store_true",
        help="Fail instead of cropping all maps to their common top-left extent.",
    )
    return parser.parse_args()


def read_frame_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    ids: list[str] = []
    for raw in path.read_text().splitlines():
        value = raw.strip()
        if not value:
            continue
        stem = Path(value).stem
        try:
            stem = f"{int(stem):06d}"
        except ValueError:
            pass
        ids.append(stem)
    if not ids:
        raise ValueError(f"split file has no frame IDs: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError("split file contains duplicate frame IDs")
    return ids


def parse_bins(text: str) -> np.ndarray:
    try:
        values = np.asarray([float(v.strip()) for v in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"invalid --depth_bins: {text}") from exc
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("--depth_bins requires at least two finite values")
    if not np.all(np.diff(values) > 0):
        raise ValueError("--depth_bins must be strictly increasing")
    return values


def load_map(root: Path, frame_id: str) -> np.ndarray:
    candidates = [root / f"{frame_id}.npy", root / frame_id]
    for path in candidates:
        if path.is_file():
            arr = np.load(path, allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"expected 2-D map, got shape {arr.shape}: {path}")
            return arr
    raise FileNotFoundError(f"missing depth map for frame {frame_id} under {root}")


def align_shapes(
    pred: np.ndarray,
    gt: np.ndarray,
    landmark: np.ndarray,
    *,
    strict: bool,
    frame_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shapes = {pred.shape, gt.shape, landmark.shape}
    if len(shapes) == 1:
        return pred, gt, landmark
    if strict:
        raise ValueError(
            f"shape mismatch for frame {frame_id}: "
            f"pred={pred.shape}, gt={gt.shape}, landmark={landmark.shape}"
        )
    height = min(pred.shape[0], gt.shape[0], landmark.shape[0])
    width = min(pred.shape[1], gt.shape[1], landmark.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"cannot align empty maps for frame {frame_id}")
    return pred[:height, :width], gt[:height, :width], landmark[:height, :width]


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else math.nan


def compute_stats(errors: np.ndarray, signed_errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {
            "median_abs_error": math.nan,
            "mean_abs_error": math.nan,
            "std_abs_error": math.nan,
            "rmse": math.nan,
            "mean_signed_error": math.nan,
            "p90_abs_error": math.nan,
            "p95_abs_error": math.nan,
        }
    return {
        "median_abs_error": safe_float(np.median(errors)),
        "mean_abs_error": safe_float(np.mean(errors)),
        "std_abs_error": safe_float(np.std(errors)),
        "rmse": safe_float(np.sqrt(np.mean(np.square(signed_errors, dtype=np.float64)))),
        "mean_signed_error": safe_float(np.mean(signed_errors)),
        "p90_abs_error": safe_float(np.percentile(errors, 90)),
        "p95_abs_error": safe_float(np.percentile(errors, 95)),
    }


def ratio(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return math.nan
    return float(value / reference)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    frame_ids = read_frame_ids(args.split_file)
    bins = parse_bins(args.depth_bins)
    num_bins = bins.size - 1

    pooled_abs: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    pooled_signed: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    contributing_frames = np.zeros(num_bins, dtype=np.int64)
    frame_rows: list[dict[str, object]] = []

    total_gt_valid = 0
    total_landmarks = 0
    total_evaluated = 0
    cropped_frame_count = 0

    for frame_id in frame_ids:
        pred = load_map(args.pred_depth_path, frame_id)
        gt = load_map(args.gt_depth_path, frame_id)
        landmark = load_map(args.landmark_depth_path, frame_id)
        original_shapes = (pred.shape, gt.shape, landmark.shape)
        pred, gt, landmark = align_shapes(
            pred, gt, landmark, strict=args.strict_shape, frame_id=frame_id
        )
        if len(set(original_shapes)) != 1:
            cropped_frame_count += 1

        gt_valid = (
            np.isfinite(gt)
            & (gt > args.gt_min)
            & (gt <= args.gt_max)
        )
        pred_valid = (
            np.isfinite(pred)
            & (pred > args.pred_min)
            & (pred <= args.pred_max)
        )
        landmark_valid = np.isfinite(landmark) & (landmark > 0.0)
        eval_valid = gt_valid & pred_valid
        if not args.include_landmarks:
            eval_valid &= ~landmark_valid

        total_gt_valid += int(np.count_nonzero(gt_valid))
        total_landmarks += int(np.count_nonzero(gt_valid & landmark_valid))
        total_evaluated += int(np.count_nonzero(eval_valid))

        signed = pred.astype(np.float64) - gt.astype(np.float64)
        absolute = np.abs(signed)

        for bin_index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            in_bin = eval_valid & (gt >= lower) & (gt < upper)
            count = int(np.count_nonzero(in_bin))
            if count:
                values_abs = absolute[in_bin].astype(np.float64, copy=False)
                values_signed = signed[in_bin].astype(np.float64, copy=False)
                pooled_abs[bin_index].append(values_abs)
                pooled_signed[bin_index].append(values_signed)
                contributing_frames[bin_index] += 1
                stats = compute_stats(values_abs, values_signed)
            else:
                stats = compute_stats(np.empty(0), np.empty(0))

            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "depth_min": float(lower),
                    "depth_max": float(upper),
                    "count": count,
                    **stats,
                }
            )

    summaries: list[BinSummary] = []
    for bin_index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        if pooled_abs[bin_index]:
            errors = np.concatenate(pooled_abs[bin_index])
            signed_errors = np.concatenate(pooled_signed[bin_index])
        else:
            errors = np.empty(0, dtype=np.float64)
            signed_errors = np.empty(0, dtype=np.float64)
        stats = compute_stats(errors, signed_errors)
        key = (float(lower), float(upper))
        paper_median = PAPER_SDN_MEDIAN.get(key, math.nan)
        paper_mean = PAPER_SDN_MEAN.get(key, math.nan)
        paper_std = PAPER_SDN_STD.get(key, math.nan)
        summaries.append(
            BinSummary(
                depth_min=float(lower),
                depth_max=float(upper),
                count=int(errors.size),
                contributing_frames=int(contributing_frames[bin_index]),
                paper_sdn_median=paper_median,
                paper_sdn_mean=paper_mean,
                paper_sdn_std=paper_std,
                median_ratio_to_paper=ratio(stats["median_abs_error"], paper_median),
                mean_ratio_to_paper=ratio(stats["mean_abs_error"], paper_mean),
                **stats,
            )
        )

    summary_rows = [asdict(item) for item in summaries]
    write_csv(args.summary_csv, summary_rows, list(summary_rows[0].keys()))
    write_csv(
        args.frame_csv,
        frame_rows,
        [
            "frame_id",
            "depth_min",
            "depth_max",
            "count",
            "median_abs_error",
            "mean_abs_error",
            "std_abs_error",
            "rmse",
            "mean_signed_error",
            "p90_abs_error",
            "p95_abs_error",
        ],
    )

    metadata = {
        "protocol": "paper_style_sdn_image_depth_non_landmark",
        "quantity": "rectified-camera-axis depth Z in meters",
        "aggregation": "pooled eligible LiDAR-projected image pixels",
        "frame_count": len(frame_ids),
        "split_file": str(args.split_file.resolve()),
        "pred_depth_path": str(args.pred_depth_path.resolve()),
        "gt_depth_path": str(args.gt_depth_path.resolve()),
        "landmark_depth_path": str(args.landmark_depth_path.resolve()),
        "depth_bins": bins.tolist(),
        "gt_valid_pixels": total_gt_valid,
        "landmark_pixels_on_gt": total_landmarks,
        "evaluated_non_landmark_pixels": total_evaluated,
        "landmark_exclusion_enabled": not args.include_landmarks,
        "cropped_shape_mismatch_frames": cropped_frame_count,
        "pred_valid_range": [args.pred_min, args.pred_max],
        "gt_valid_range": [args.gt_min, args.gt_max],
        "summary_csv": str(args.summary_csv.resolve()),
        "frame_csv": str(args.frame_csv.resolve()),
    }
    metadata_path = args.metadata_json or args.summary_csv.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nPaper-style SDN image-depth sanity check")
    print(f"frames                  : {len(frame_ids)}")
    print(f"GT-valid pixels         : {total_gt_valid:,}")
    print(f"excluded landmark pixels: {total_landmarks:,}")
    print(f"evaluated pixels        : {total_evaluated:,}")
    print(f"shape-cropped frames    : {cropped_frame_count}")
    print()
    header = (
        f"{'Depth (m)':>11}  {'Count':>10}  {'Median':>9}  {'Paper':>8}  "
        f"{'Ratio':>7}  {'Mean':>9}  {'Paper':>8}  {'RMSE':>9}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        interval = f"{item.depth_min:g}-{item.depth_max:g}"
        print(
            f"{interval:>11}  {item.count:10,d}  "
            f"{item.median_abs_error:9.4f}  {item.paper_sdn_median:8.4f}  "
            f"{item.median_ratio_to_paper:7.2f}  {item.mean_abs_error:9.4f}  "
            f"{item.paper_sdn_mean:8.4f}  {item.rmse:9.4f}"
        )
    print(f"\nsummary : {args.summary_csv}")
    print(f"per-frame: {args.frame_csv}")
    print(f"metadata : {metadata_path}")


if __name__ == "__main__":
    main()
