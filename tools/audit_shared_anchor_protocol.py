#!/usr/bin/env python3
"""Verify values, identity, and provenance of canonical GDC/RGC anchors."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from range_gdc.shared_canonical_anchor import sha256_file, shared_points_to_camera_depth

sys.path.insert(0, str(REPO_ROOT / "gdc"))
from data_utils.kitti_util import Calibration, load_image  # noqa: E402


DEFAULT_ATOL = 1e-5
FAILURE_COUNT_FIELDS = (
    "shared_sparse_sha256_mismatch_count",
    "shared_point_count_mismatch_count",
    "rgc_source_index_grid_mismatch_count",
    "rgc_anchor_not_from_shared_count",
    "rgc_range_value_mismatch_count",
    "gdc_anchor_not_from_shared_count",
    "gdc_depth_value_mismatch_count",
)


def ids(path):
    with open(path) as handle:
        return [f"{int(line.strip()):06d}" for line in handle if line.strip()]


def load_points(path):
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path}: malformed x/y/z/intensity records")
    return values.reshape(-1, 4)


def _ratio(bad, total):
    return float(bad / total) if total else 0.0


def _value_comparison(actual, expected, atol):
    """Compare validity and values, returning global mismatch/error metrics."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"actual shape {actual.shape} != expected shape {expected.shape}")
    actual_valid = np.isfinite(actual) & (actual > 0)
    expected_valid = np.isfinite(expected) & (expected > 0)
    both = actual_valid & expected_valid
    comparable_error = np.abs(actual[both].astype(np.float64) - expected[both].astype(np.float64))
    value_bad = np.zeros(actual.shape, dtype=bool)
    if comparable_error.size:
        value_bad[both] = comparable_error > float(atol)
    mismatch = (actual_valid ^ expected_valid) | value_bad
    union_count = int((actual_valid | expected_valid).sum())
    mismatch_count = int(mismatch.sum())
    mismatched_errors = comparable_error[comparable_error > float(atol)]
    return {
        "actual_valid_count": int(actual_valid.sum()),
        "expected_valid_count": int(expected_valid.sum()),
        "comparison_valid_count": union_count,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": _ratio(mismatch_count, union_count),
        "mae": float(np.mean(comparable_error)) if comparable_error.size else 0.0,
        "max_abs_error": float(np.max(comparable_error)) if comparable_error.size else 0.0,
        "mismatched_value_mae": float(np.mean(mismatched_errors)) if mismatched_errors.size else 0.0,
        "actual_valid": actual_valid,
        "expected_valid": expected_valid,
    }


def expected_rgc_from_shared(points, source_index, selected_rows, invalid_value=0.0):
    """Rebuild RGC range strictly from sorted selected source IDs and shared PCD."""
    points = np.asarray(points, dtype=np.float32)
    source_index = np.asarray(source_index, dtype=np.int32)
    rows = np.asarray(selected_rows, dtype=np.int32)
    if source_index.ndim != 2 or np.any(rows < 0) or np.any(rows >= source_index.shape[0]):
        raise ValueError("selected rows are outside source-index grid")
    selected = source_index[rows]
    unique_indices = np.unique(selected[selected >= 0])
    unique_indices.sort()
    expected = np.full(source_index.shape, float(invalid_value), dtype=np.float32)
    if unique_indices.size != points.shape[0]:
        return expected, unique_indices
    point_ranges = np.linalg.norm(points[:, :3], axis=1).astype(np.float32)
    lookup = dict(zip(unique_indices.tolist(), point_ranges.tolist()))
    selected_expected = expected[rows]
    valid = selected >= 0
    selected_expected[valid] = np.asarray(
        [lookup[int(index)] for index in selected[valid]], dtype=np.float32
    )
    expected[rows] = selected_expected
    return expected, unique_indices


def audit_frame(
    frame_id, points, source_index, recorded_rgc_source_index, rgc_anchor,
    actual_gdc_depth, expected_gdc_depth, manifest_frame, actual_shared_sha256,
    selected_rows, atol=DEFAULT_ATOL,
):
    """Audit one frame without file I/O, suitable for CLI and unit tests."""
    points = np.asarray(points, dtype=np.float32)
    source_index = np.asarray(source_index, dtype=np.int32)
    recorded = np.asarray(recorded_rgc_source_index, dtype=np.int32)
    if recorded.shape != source_index.shape:
        raise ValueError(f"{frame_id}: RGC source-index shape differs from canonical grid")
    source_grid_match = bool(np.array_equal(recorded, source_index))
    expected_rgc, unique_indices = expected_rgc_from_shared(points, source_index, selected_rows)
    manifest_count = int(manifest_frame.get("shared_sparse_point_count", -1))
    point_count = int(points.shape[0])
    point_count_match = bool(unique_indices.size == point_count == manifest_count)
    expected_sha = str(manifest_frame.get("sha256", ""))
    sha_match = bool(expected_sha and expected_sha == str(actual_shared_sha256))

    rgc_compare = _value_comparison(rgc_anchor, expected_rgc, atol)
    rows_mask = np.zeros(source_index.shape[0], dtype=bool)
    rows_mask[np.asarray(selected_rows, dtype=np.int32)] = True
    rgc_not_shared_mask = rgc_compare["actual_valid"] & (
        (source_index < 0) | ~rows_mask[:, None] | (not source_grid_match)
    )
    rgc_not_shared = int(rgc_not_shared_mask.sum())

    gdc_compare = _value_comparison(actual_gdc_depth, expected_gdc_depth, atol)
    gdc_not_shared = int((gdc_compare["actual_valid"] & ~gdc_compare["expected_valid"]).sum())

    return {
        "frame_id": str(frame_id),
        "shared_sparse_manifest_sha256": expected_sha,
        "shared_sparse_actual_sha256": str(actual_shared_sha256),
        "shared_sparse_sha256_match": sha_match,
        "shared_sparse_sha256_mismatch_count": int(not sha_match),
        "selected_unique_source_index_count": int(unique_indices.size),
        "shared_sparse_point_count": point_count,
        "manifest_shared_sparse_point_count": manifest_count,
        "shared_point_count_match": point_count_match,
        "shared_point_count_mismatch_count": int(not point_count_match),
        "rgc_source_index_grid_match": source_grid_match,
        "rgc_source_index_grid_mismatch_count": int(not source_grid_match),
        "rgc_anchor_valid_count": rgc_compare["actual_valid_count"],
        "rgc_expected_valid_count": rgc_compare["expected_valid_count"],
        "rgc_comparison_valid_count": rgc_compare["comparison_valid_count"],
        "rgc_anchor_not_from_shared_count": rgc_not_shared,
        "rgc_anchor_not_from_shared_ratio": _ratio(rgc_not_shared, rgc_compare["actual_valid_count"]),
        "rgc_range_value_mismatch_count": rgc_compare["mismatch_count"],
        "rgc_range_value_mismatch_ratio": rgc_compare["mismatch_ratio"],
        "rgc_range_value_mae": rgc_compare["mae"],
        "rgc_range_value_max_abs_error": rgc_compare["max_abs_error"],
        "gdc_projected_shared_point_count": gdc_compare["expected_valid_count"],
        "gdc_anchor_valid_count": gdc_compare["actual_valid_count"],
        "gdc_comparison_valid_count": gdc_compare["comparison_valid_count"],
        "gdc_anchor_not_from_shared_count": gdc_not_shared,
        "gdc_anchor_not_from_shared_ratio": _ratio(gdc_not_shared, gdc_compare["actual_valid_count"]),
        "gdc_depth_value_mismatch_count": gdc_compare["mismatch_count"],
        "gdc_depth_value_mismatch_ratio": gdc_compare["mismatch_ratio"],
        "gdc_depth_value_max_abs_error": gdc_compare["max_abs_error"],
    }


def aggregate_audit(rows):
    """Aggregate per-frame metrics with real global denominators."""
    total = lambda key: int(sum(int(row[key]) for row in rows))
    rgc_valid = total("rgc_anchor_valid_count")
    rgc_comparison = total("rgc_comparison_valid_count")
    gdc_valid = total("gdc_anchor_valid_count")
    gdc_comparison = total("gdc_comparison_valid_count")
    rgc_external = total("rgc_anchor_not_from_shared_count")
    rgc_value_bad = total("rgc_range_value_mismatch_count")
    gdc_external = total("gdc_anchor_not_from_shared_count")
    gdc_value_bad = total("gdc_depth_value_mismatch_count")
    return {
        "frame_count": len(rows),
        "shared_sparse_sha256_mismatch_count": total("shared_sparse_sha256_mismatch_count"),
        "shared_point_count_mismatch_count": total("shared_point_count_mismatch_count"),
        "rgc_source_index_grid_mismatch_count": total("rgc_source_index_grid_mismatch_count"),
        "rgc_anchor_not_from_shared_count": rgc_external,
        "rgc_anchor_not_from_shared_ratio": _ratio(rgc_external, rgc_valid),
        "rgc_range_value_mismatch_count": rgc_value_bad,
        "rgc_range_value_mismatch_ratio": _ratio(rgc_value_bad, rgc_comparison),
        "rgc_range_value_max_abs_error": max((float(row["rgc_range_value_max_abs_error"]) for row in rows), default=0.0),
        "gdc_anchor_not_from_shared_count": gdc_external,
        "gdc_anchor_not_from_shared_ratio": _ratio(gdc_external, gdc_valid),
        "gdc_depth_value_mismatch_count": gdc_value_bad,
        "gdc_depth_value_mismatch_ratio": _ratio(gdc_value_bad, gdc_comparison),
        "gdc_depth_value_max_abs_error": max((float(row["gdc_depth_value_max_abs_error"]) for row in rows), default=0.0),
    }


def write_frame_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["frame_id"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in summary.items())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")
    root = Path(args.output_root)
    pc_dir = root / "anchor" / "shared_canonical_pointcloud"
    source_dir = root / "anchor" / "shared_canonical_source_index"
    manifest_path = root / "anchor" / "shared_canonical_pointcloud_provenance.json"
    rgc_dir = root / "anchor" / "range_shared_canonical" / "G64_range"
    rgc_meta = root / "anchor" / "range_shared_canonical" / "meta"
    gdc_dir = root / "anchor" / "shared_canonical_image_depth"
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest_sha = sha256_file(manifest_path)
    with open(rgc_meta / "anchor_definition.json") as handle:
        rgc_definition = json.load(handle)
    with open(gdc_dir / "anchor_provenance.json") as handle:
        gdc_definition = json.load(handle)
    if (rgc_definition.get("source_manifest_sha256") != manifest_sha
            or gdc_definition.get("source_manifest_sha256") != manifest_sha):
        raise ValueError("GDC/RGC provenance does not reference the same shared manifest")
    selected_rows = np.asarray(manifest["selected_rows"], dtype=np.int32)
    frame_manifest = {str(row["frame_id"]): row for row in manifest["frames"]}
    rows = []
    for stem in ids(args.split_file):
        if stem not in frame_manifest:
            raise ValueError(f"{stem}: missing frame entry in shared manifest")
        pc_path = pc_dir / f"{stem}.bin"
        points = load_points(pc_path)
        source = np.load(source_dir / f"{stem}.npy")
        recorded_source = np.load(rgc_meta / f"{stem}_source_index.npy")
        rgc = np.load(rgc_dir / f"{stem}.npy")
        calib = Calibration(str(Path(args.calib_dir) / f"{stem}.txt"))
        image = load_image(str(Path(args.image_dir) / f"{stem}.png"))
        expected_gdc = shared_points_to_camera_depth(points, calib, image.shape)
        actual_gdc = np.load(gdc_dir / f"{stem}.npy")
        rows.append(audit_frame(
            stem, points, source, recorded_source, rgc, actual_gdc, expected_gdc,
            frame_manifest[stem], sha256_file(pc_path), selected_rows, args.atol,
        ))

    output = Path(args.output_csv or root / "metrics" / "shared_anchor_protocol_audit.csv")
    summary_path = Path(args.summary_csv or root / "metrics" / "shared_anchor_protocol_summary.csv")
    summary = aggregate_audit(rows)
    write_frame_csv(output, rows)
    write_summary_csv(summary_path, summary)
    failures = {key: summary[key] for key in FAILURE_COUNT_FIELDS if summary[key] != 0}
    if failures:
        raise SystemExit(f"Shared-anchor audit failed: {failures}")
    print(f"Saved shared-anchor audit: {output}")
    print(f"Saved shared-anchor summary: {summary_path}")


if __name__ == "__main__":
    main()
