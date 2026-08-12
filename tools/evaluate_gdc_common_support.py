#!/usr/bin/env python3
"""Evaluate methods on output-independent pre-correction GDC support.

The support mask is derived only from Raw SDN camera depth, KITTI calibration,
production GDC geometric eligibility, and canonical spherical projection. It is
not an activity/changed mask and is not exact corrected-GDC node correspondence.
No correction method is run by this offline diagnostic.
"""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GDC_ROOT = REPO_ROOT / "gdc"
if str(GDC_ROOT) not in sys.path:
    sys.path.insert(0, str(GDC_ROOT))

from data_utils.kitti_util import Calibration  # noqa: E402
from gdc import depth2ptc, filter_mask, filter_theta_mask  # noqa: E402
from range_gdc.evaluate_range_metrics import error_metrics  # noqa: E402
from range_gdc.range_main_batch import npy_map, read_split_scene_ids  # noqa: E402
from src.pseudo_lidar.depth_to_range_uniform import (  # noqa: E402
    azimuth_to_col_uniform,
    elevation_to_row_uniform,
)
from tools.analyze_boundary_topology import (  # noqa: E402
    DISTANCE_BINS,
    boundary_distance_bin_mask,
    common_hidden_valid_mask,
    gt_boundary_mask,
    hidden_rows_mask,
    periodic_boundary_distance,
    region_mask,
    resolve_projection_meta,
    validate_output_dir,
    write_csv,
)


SUPPORT_REGIONS = (
    "all_common_support",
    "boundary_common_support",
    "interior_common_support",
)
DEFAULT_CONSIDER_RANGE = (-0.1, 3.0)
DEFAULT_MAPPING_TOL = 1e-5


def projection_parameters(projection):
    """Return production uniform-grid parameters from projection metadata."""
    height, width = projection["shape"]
    vertical = np.asarray(projection["vertical_centers_deg"], dtype=np.float64)
    azimuth = np.asarray(projection["azimuth_centers_deg"], dtype=np.float64)
    if height < 1 or width < 1:
        raise ValueError("projection grid must be non-empty")
    v_step = (
        abs(float(vertical[0] - vertical[1]))
        if height > 1 else float(projection.get("vertical_resolution_deg", 1.0))
    )
    a_step = (
        abs(float(azimuth[1] - azimuth[0]))
        if width > 1 else float(projection.get("horizontal_resolution_deg", 360.0))
    )
    return {
        "range_h": height,
        "range_w": width,
        "vmin_deg": float(projection.get("vmin_deg", vertical[-1] - 0.5 * v_step)),
        "vmax_deg": float(projection.get("vmax_deg", vertical[0] + 0.5 * v_step)),
        "azimuth_mode": projection["azimuth_mode"],
        "azimuth_min_deg": float(
            projection.get("azimuth_min_deg", azimuth[0] - 0.5 * a_step)
        ),
        "azimuth_max_deg": float(
            projection.get("azimuth_max_deg", azimuth[-1] + 0.5 * a_step)
        ),
    }


def canonical_winner_map(points_velo, source_flat_indices, projection,
                         range_min=0.1, range_max=80.0, invalid_value=0.0):
    """Project all points and retain nearest range plus its source pixel.

    Ranges are cast to float32 before collision resolution, matching production
    ``np.minimum.at``. Equal float32 ranges choose the smallest source flat
    camera-pixel index deterministically.
    """
    if range_max <= range_min:
        raise ValueError("range_max must be greater than range_min")
    points = np.asarray(points_velo, dtype=np.float32)
    source = np.asarray(source_flat_indices, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] != source.size:
        raise ValueError("points and source_flat_indices have incompatible shapes")
    parameters = projection_parameters(projection)
    shape = (parameters["range_h"], parameters["range_w"])
    reconstructed = np.full(shape, float(invalid_value), dtype=np.float32)
    winners = np.full(shape, -1, dtype=np.int64)
    if not source.size:
        return reconstructed, winners

    xyz = points[:, :3].astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ranges = np.linalg.norm(xyz, axis=1)
    horizontal = np.sqrt(x * x + y * y)
    valid = (
        np.isfinite(ranges)
        & (ranges > float(range_min))
        & (ranges < float(range_max))
        & (horizontal > 0.0)
    )
    if not np.any(valid):
        return reconstructed, winners
    ranges = ranges[valid]
    source = source[valid]
    elevation = np.degrees(np.arctan2(z[valid], horizontal[valid]))
    azimuth = np.degrees(np.arctan2(y[valid], x[valid]))
    rows = elevation_to_row_uniform(
        elevation,
        parameters["range_h"],
        parameters["vmin_deg"],
        parameters["vmax_deg"],
    )
    cols = azimuth_to_col_uniform(
        azimuth,
        parameters["range_w"],
        parameters["azimuth_mode"],
        parameters["azimuth_min_deg"],
        parameters["azimuth_max_deg"],
    )
    inside = (
        (rows >= 0)
        & (rows < parameters["range_h"])
        & (cols >= 0)
        & (cols < parameters["range_w"])
    )
    rows = rows[inside]
    cols = cols[inside]
    ranges = ranges[inside].astype(np.float32)
    source = source[inside]
    if not source.size:
        return reconstructed, winners

    cells = rows * parameters["range_w"] + cols
    order = np.lexsort((source, ranges, cells))
    sorted_cells = cells[order]
    first = np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
    selected = order[first]
    flat_reconstructed = reconstructed.reshape(-1)
    flat_winners = winners.reshape(-1)
    flat_reconstructed[cells[selected]] = ranges[selected]
    flat_winners[cells[selected]] = source[selected]
    return reconstructed, winners


def raw_depth_to_canonical_winner_map(raw_depth, calib, projection,
                                      range_min=0.1, range_max=80.0):
    """Reconstruct production Raw SDN range and original camera winners."""
    depth = np.asarray(raw_depth, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth > float(range_min))
        & (depth < float(range_max))
    )
    rows, cols = np.where(valid)
    if not rows.size:
        return canonical_winner_map(
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            projection,
            range_min,
            range_max,
        )
    z = depth[rows, cols].astype(np.float32)
    uv_depth = np.column_stack(
        (cols.astype(np.float32), rows.astype(np.float32), z)
    )
    points_rect = calib.project_image_to_rect(uv_depth)
    points_velo = calib.project_rect_to_velo(points_rect).astype(np.float32)
    source = np.ravel_multi_index((rows, cols), depth.shape).astype(np.int64)
    return canonical_winner_map(
        points_velo, source, projection, range_min, range_max
    )


def production_gdc_eligibility(raw_depth, calib,
                               consider_range=DEFAULT_CONSIDER_RANGE):
    """Compute production naive-GDC ``consider_PL`` without any anchor."""
    if len(consider_range) != 2 or consider_range[1] <= consider_range[0]:
        raise ValueError("consider_range must contain increasing low/high degrees")
    points_rect = depth2ptc(np.asarray(raw_depth), calib)
    return (
        filter_mask(points_rect)
        & filter_theta_mask(
            points_rect,
            low=np.radians(float(consider_range[0])),
            high=np.radians(float(consider_range[1])),
        )
    ).reshape(np.asarray(raw_depth).shape)


def gdc_eligible_canonical_mask(winner_flat_pixel_index, consider_pl):
    winners = np.asarray(winner_flat_pixel_index, dtype=np.int64)
    eligibility = np.asarray(consider_pl, dtype=bool).reshape(-1)
    output = np.zeros(winners.shape, dtype=bool)
    valid = winners >= 0
    if np.any(winners[valid] >= eligibility.size):
        raise ValueError("winner index is outside the camera depth image")
    output[valid] = eligibility[winners[valid]]
    return output


def common_gdc_support_mask(common_hidden_valid, eligible_canonical):
    common = np.asarray(common_hidden_valid, dtype=bool)
    eligible = np.asarray(eligible_canonical, dtype=bool)
    if common.shape != eligible.shape:
        raise ValueError("common and eligible masks must share a shape")
    return common & eligible


def audit_reconstructed_raw_range(reconstructed, artifact, tolerance=DEFAULT_MAPPING_TOL,
                                  fail_on_mismatch=True):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("mapping tolerance must be finite and nonnegative")
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    artifact = np.asarray(artifact, dtype=np.float32)
    if reconstructed.shape != artifact.shape:
        raise ValueError("reconstructed and artifact range shapes differ")
    reconstructed_valid = np.isfinite(reconstructed) & (reconstructed > 0)
    artifact_valid = np.isfinite(artifact) & (artifact > 0)
    valid_mismatch = reconstructed_valid ^ artifact_valid
    common = reconstructed_valid & artifact_valid
    absolute_difference = np.abs(
        reconstructed[common].astype(np.float64) - artifact[common].astype(np.float64)
    )
    range_mismatch = absolute_difference > float(tolerance)
    row = {
        "reconstructed_valid_pixels": int(reconstructed_valid.sum()),
        "artifact_valid_pixels": int(artifact_valid.sum()),
        "valid_mask_mismatch_count": int(valid_mismatch.sum()),
        "range_mismatch_count": int(range_mismatch.sum()),
        "mismatched_range_cell_count": int(valid_mismatch.sum() + range_mismatch.sum()),
        "max_absolute_range_difference": (
            float(np.max(absolute_difference)) if absolute_difference.size else 0.0
        ),
        "mapping_tolerance": float(tolerance),
    }
    row["status"] = "ok" if row["mismatched_range_cell_count"] == 0 else "mismatch"
    if fail_on_mismatch and row["status"] != "ok":
        raise ValueError(
            "Raw range reconstruction mismatch: "
            f"valid={row['valid_mask_mismatch_count']} "
            f"range={row['range_mismatch_count']} "
            f"max_abs={row['max_absolute_range_difference']}"
        )
    return row


def support_region_mask(common, support, distance, region, boundary_radius):
    if region == "all_common_support":
        return support
    if region == "boundary_common_support":
        return support & region_mask(common, distance, "boundary", boundary_radius)
    if region == "interior_common_support":
        return support & region_mask(common, distance, "interior", boundary_radius)
    raise ValueError(f"unknown support region: {region}")


def base_region_mask(common, distance, region, boundary_radius):
    if region == "all_common_support":
        return common
    if region == "boundary_common_support":
        return region_mask(common, distance, "boundary", boundary_radius)
    if region == "interior_common_support":
        return region_mask(common, distance, "interior", boundary_radius)
    raise ValueError(f"unknown support region: {region}")


def method_error_row(prediction, gt, mask):
    errors = np.asarray(prediction, dtype=np.float64)[mask] - np.asarray(
        gt, dtype=np.float64
    )[mask]
    return {"pixels": int(errors.size), **error_metrics(errors)}, errors


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _input_maps(args, scene_ids):
    directories = {
        "raw_sdn": args.raw_range_dir,
        "gt": args.gt_range_dir,
        "range_gdc": args.rgc_range_dir,
        "original_gdc": args.gdc_range_dir,
        "raw_depth": args.raw_depth_dir,
    }
    mappings = {}
    for name, directory in directories.items():
        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(path)
        mapping = npy_map(str(path))
        missing = [scene_id for scene_id in scene_ids if scene_id not in mapping]
        if missing:
            raise FileNotFoundError(
                f"{name} missing {len(missing)} frames, sample={missing[:5]}"
            )
        mappings[name] = mapping
    calib_dir = Path(args.calib_dir)
    if not calib_dir.is_dir():
        raise FileNotFoundError(calib_dir)
    for scene_id in scene_ids:
        if not (calib_dir / f"{scene_id}.txt").is_file():
            raise FileNotFoundError(calib_dir / f"{scene_id}.txt")
    return directories, mappings


def evaluate(args):
    projection = resolve_projection_meta(args)
    source_rows = args.source_rows or projection["selected_rows"] or [5, 7, 9, 11]
    source_rows = [int(value) for value in source_rows]
    scene_ids = read_split_scene_ids(args.split_file)
    if args.max_items is not None:
        scene_ids = scene_ids[:args.max_items]
    directories, mappings = _input_maps(args, scene_ids)
    output_dir = validate_output_dir(
        args.output_dir, [*directories.values(), args.calib_dir]
    )
    method_specs = (
        ("raw_sdn", "raw_sdn"),
        (args.gdc_label, "original_gdc"),
        ("range_gdc", "range_gdc"),
    )

    audit_rows = []
    prepared = []
    for scene_id in scene_ids:
        raw_depth = np.load(mappings["raw_depth"][scene_id])
        raw_range = np.load(mappings["raw_sdn"][scene_id]).astype(np.float32)
        if raw_range.shape != projection["shape"]:
            raise ValueError(
                f"{scene_id} raw_sdn: {raw_range.shape} != {projection['shape']}"
            )
        calib = Calibration(str(Path(args.calib_dir) / f"{scene_id}.txt"))
        reconstructed, winners = raw_depth_to_canonical_winner_map(
            raw_depth, calib, projection, args.range_min, args.range_max
        )
        audit = audit_reconstructed_raw_range(
            reconstructed, raw_range, args.mapping_tol, fail_on_mismatch=False
        )
        audit_rows.append({"frame_id": scene_id, **audit})
        prepared.append((scene_id, raw_depth, winners, calib))

    mismatched_frames = [row for row in audit_rows if row["status"] != "ok"]
    mapping_summary = {
        "frames": len(scene_ids),
        "frames_ok": len(scene_ids) - len(mismatched_frames),
        "frames_mismatched": len(mismatched_frames),
        "reconstructed_valid_pixels": sum(
            row["reconstructed_valid_pixels"] for row in audit_rows
        ),
        "artifact_valid_pixels": sum(row["artifact_valid_pixels"] for row in audit_rows),
        "valid_mask_mismatch_count": sum(
            row["valid_mask_mismatch_count"] for row in audit_rows
        ),
        "range_mismatch_count": sum(row["range_mismatch_count"] for row in audit_rows),
        "mismatched_range_cell_count": sum(
            row["mismatched_range_cell_count"] for row in audit_rows
        ),
        "max_absolute_range_difference": max(
            (row["max_absolute_range_difference"] for row in audit_rows), default=0.0
        ),
        "mapping_tolerance": args.mapping_tol,
        "status": "mismatch" if mismatched_frames else "ok",
    }
    write_csv(output_dir / "common_gdc_support_mapping_audit.csv", audit_rows)
    write_csv(
        output_dir / "common_gdc_support_mapping_summary.csv", [mapping_summary]
    )
    if mismatched_frames:
        sample = mismatched_frames[0]
        raise ValueError(
            "Raw range reconstruction audit failed before support outputs: "
            f"{len(mismatched_frames)} frame(s), first={sample['frame_id']}, "
            f"cells={sample['mismatched_range_cell_count']}, "
            f"max_abs={sample['max_absolute_range_difference']}"
        )

    per_frame_rows = []
    summary_acc = defaultdict(lambda: {"errors": [], "common": 0, "eligible": 0, "support": 0})
    distance_acc = defaultdict(lambda: {"errors": [], "common": 0, "eligible": 0, "support": 0})
    for scene_id, raw_depth, winners, calib in prepared:
        arrays = {
            name: np.load(mapping[scene_id]).astype(np.float32)
            for name, mapping in mappings.items() if name != "raw_depth"
        }
        for name, array in arrays.items():
            if array.shape != projection["shape"]:
                raise ValueError(
                    f"{scene_id} {name}: {array.shape} != {projection['shape']}"
                )
        predictions = [arrays[array_name] for _, array_name in method_specs]
        common = common_hidden_valid_mask(
            arrays["gt"], predictions, source_rows, args.range_min, args.range_max
        )
        consider_pl = production_gdc_eligibility(
            raw_depth, calib, args.consider_range
        )
        eligible = gdc_eligible_canonical_mask(winners, consider_pl)
        support = common_gdc_support_mask(common, eligible)
        boundary = gt_boundary_mask(
            arrays["gt"], args.range_min, args.range_max, args.boundary_log_thr
        )
        distance = periodic_boundary_distance(boundary)
        hidden = hidden_rows_mask(common.shape, source_rows)

        for method, array_name in method_specs:
            for region in SUPPORT_REGIONS:
                base = base_region_mask(
                    common, distance, region, args.boundary_radius
                )
                mask = support_region_mask(
                    common, support, distance, region, args.boundary_radius
                )
                eligible_region = eligible & hidden
                if region == "boundary_common_support":
                    eligible_region &= distance <= float(args.boundary_radius)
                elif region == "interior_common_support":
                    eligible_region &= distance >= 4.0
                metrics, errors = method_error_row(
                    arrays[array_name], arrays["gt"], mask
                )
                coverage = {
                    "common_hidden_valid_pixels": int(base.sum()),
                    "gdc_eligible_pixels": int(eligible_region.sum()),
                    "common_gdc_support_pixels": int(mask.sum()),
                    "gdc_support_ratio_of_common_hidden": _ratio(
                        int(mask.sum()), int(base.sum())
                    ),
                }
                per_frame_rows.append({
                    "frame_id": scene_id,
                    "method": method,
                    "region": region,
                    **coverage,
                    **metrics,
                })
                item = summary_acc[(method, region)]
                item["common"] += coverage["common_hidden_valid_pixels"]
                item["eligible"] += coverage["gdc_eligible_pixels"]
                item["support"] += coverage["common_gdc_support_pixels"]
                if errors.size:
                    item["errors"].append(errors)

            for label in DISTANCE_BINS:
                base = boundary_distance_bin_mask(common, distance, label)
                mask = boundary_distance_bin_mask(support, distance, label)
                eligible_bin = boundary_distance_bin_mask(
                    eligible & hidden, distance, label
                )
                _, errors = method_error_row(
                    arrays[array_name], arrays["gt"], mask
                )
                item = distance_acc[(method, label)]
                item["common"] += int(base.sum())
                item["eligible"] += int(eligible_bin.sum())
                item["support"] += int(mask.sum())
                if errors.size:
                    item["errors"].append(errors)

    summary_rows = []
    for (method, region), item in sorted(summary_acc.items()):
        errors = np.concatenate(item["errors"]) if item["errors"] else np.array([])
        summary_rows.append({
            "method": method,
            "region": region,
            "frames": len(scene_ids),
            "common_hidden_valid_pixels": item["common"],
            "gdc_eligible_pixels": item["eligible"],
            "common_gdc_support_pixels": item["support"],
            "gdc_support_ratio_of_common_hidden": _ratio(
                item["support"], item["common"]
            ),
            "pixels": int(errors.size),
            **error_metrics(errors),
        })

    order = {label: index for index, label in enumerate(DISTANCE_BINS)}
    distance_rows = []
    for (method, label), item in sorted(
        distance_acc.items(), key=lambda value: (value[0][0], order[value[0][1]])
    ):
        errors = np.concatenate(item["errors"]) if item["errors"] else np.array([])
        distance_rows.append({
            "method": method,
            "boundary_distance_bin": label,
            "frames": len(scene_ids),
            "common_hidden_valid_pixels": item["common"],
            "gdc_eligible_pixels": item["eligible"],
            "common_gdc_support_pixels": item["support"],
            "gdc_support_ratio_of_common_hidden": _ratio(
                item["support"], item["common"]
            ),
            "pixels": int(errors.size),
            **error_metrics(errors),
        })

    write_csv(output_dir / "common_gdc_support_per_frame.csv", per_frame_rows)
    write_csv(output_dir / "common_gdc_support_summary.csv", summary_rows)
    write_csv(
        output_dir / "common_gdc_support_distance_summary.csv", distance_rows
    )
    metadata = {
        "analysis_kind": "offline_pre_correction_gdc_geometric_support_evaluation",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "number_of_frames": len(scene_ids),
        "split_file": str(Path(args.split_file).resolve()),
        "source_rows": source_rows,
        "range_min": args.range_min,
        "range_max": args.range_max,
        "consider_range_degrees": list(args.consider_range),
        "gdc_subsample": False,
        "gdc_label": args.gdc_label,
        "boundary_log_thr": args.boundary_log_thr,
        "boundary_radius": args.boundary_radius,
        "mapping_tolerance": args.mapping_tol,
        "winner_tie_policy": "minimum float32 spherical range, then smallest flat camera pixel index",
        "support_definition": "common_hidden_valid AND canonical Raw winner production-GDC consider_PL eligibility",
        "support_semantics": "output-independent pre-correction geometric eligibility support; not activity and not exact corrected-GDC node correspondence",
        "activity_distinction": "gdc_support_ratio describes pre-correction eligibility; changed_ratio describes post-correction output activity",
        "coverage_field_semantics": {
            "common_hidden_valid_pixels": "shared valid evaluation pixels in the named all/boundary/interior region",
            "gdc_eligible_pixels": "eligible canonical Raw winners after source-row exclusion in the named region, before common validity intersection",
            "common_gdc_support_pixels": "intersection of common_hidden_valid and GDC eligibility in the named region",
            "gdc_support_ratio_of_common_hidden": "common_gdc_support_pixels / common_hidden_valid_pixels",
        },
        "input_directories": {
            **{name: str(Path(path).resolve()) for name, path in directories.items()},
            "calibration": str(Path(args.calib_dir).resolve()),
        },
        "projection_meta_path": projection["path"],
        "mapping_audit": mapping_summary,
    }
    with (output_dir / "common_gdc_support_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    print(f"Evaluated {len(scene_ids)} frame(s) on common GDC support")
    print("Raw range reconstruction audit: 0 mismatched cells")
    print(f"Saved: {output_dir}")
    return {
        "summary": summary_rows,
        "distance_summary": distance_rows,
        "mapping_summary": mapping_summary,
        "metadata": metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-depth-dir", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--raw-range-dir", required=True)
    parser.add_argument("--gt-range-dir", required=True)
    parser.add_argument("--rgc-range-dir", required=True)
    parser.add_argument("--gdc-range-dir", required=True)
    parser.add_argument("--gdc-label", default="original_gdc_naive")
    meta = parser.add_mutually_exclusive_group(required=True)
    meta.add_argument("--projection-meta-path")
    meta.add_argument("--meta-dir")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rows", type=int, nargs="+", default=None)
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument(
        "--consider-range", type=float, nargs=2,
        default=list(DEFAULT_CONSIDER_RANGE), metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--mapping-tol", type=float, default=DEFAULT_MAPPING_TOL)
    parser.add_argument("--boundary-log-thr", type=float, default=0.2)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()
    if args.range_max <= args.range_min:
        parser.error("--range-max must be greater than --range-min")
    if args.consider_range[1] <= args.consider_range[0]:
        parser.error("--consider-range HIGH must be greater than LOW")
    if not math.isfinite(args.mapping_tol) or args.mapping_tol < 0:
        parser.error("--mapping-tol must be finite and nonnegative")
    if args.boundary_log_thr < 0 or args.boundary_radius < 0:
        parser.error("boundary threshold/radius must be nonnegative")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    args.gdc_label = args.gdc_label.strip()
    if not args.gdc_label or args.gdc_label in {"raw_sdn", "range_gdc"}:
        parser.error("--gdc-label must be non-empty and unique")
    return args


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
