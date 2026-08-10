#!/usr/bin/env python3
"""Verify that canonical GDC and RGC anchors consume one shared point source."""

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


def ids(path):
    with open(path) as handle:
        return [f"{int(line.strip()):06d}" for line in handle if line.strip()]


def load_points(path):
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path}: malformed x/y/z/intensity records")
    return values.reshape(-1, 4)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
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
    if rgc_definition.get("source_manifest_sha256") != manifest_sha or gdc_definition.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("GDC/RGC provenance does not reference the same shared manifest")
    selected_rows = np.asarray(manifest["selected_rows"], dtype=np.int32)
    frame_manifest = {row["frame_id"]: row for row in manifest["frames"]}
    rows = []
    for stem in ids(args.split_file):
        points = load_points(pc_dir / f"{stem}.bin")
        source = np.load(source_dir / f"{stem}.npy")
        rgc = np.load(rgc_dir / f"{stem}.npy")
        recorded_source = np.load(rgc_meta / f"{stem}_source_index.npy")
        if not np.array_equal(source, recorded_source):
            raise ValueError(f"{stem}: RGC uses a different source-index grid")
        rgc_valid = np.isfinite(rgc) & (rgc > 0)
        outside = np.ones(source.shape[0], dtype=bool); outside[selected_rows] = False
        rgc_not_shared = int(np.sum(rgc_valid & ((source < 0) | outside[:, None])))
        calib = Calibration(str(Path(args.calib_dir) / f"{stem}.txt"))
        image = load_image(str(Path(args.image_dir) / f"{stem}.png"))
        expected_gdc = shared_points_to_camera_depth(points, calib, image.shape)
        actual_gdc = np.load(gdc_dir / f"{stem}.npy")
        gdc_valid = actual_gdc > 0
        expected_valid = expected_gdc > 0
        gdc_not_shared = int(np.sum(gdc_valid & (~expected_valid | ~np.isclose(actual_gdc, expected_gdc))))
        rows.append({
            "frame_id": stem, "shared_sparse_point_count": int(points.shape[0]),
            "shared_sparse_sha256": frame_manifest[stem]["sha256"],
            "rgc_anchor_valid_count": int(rgc_valid.sum()),
            "rgc_anchor_not_from_shared_count": rgc_not_shared,
            "rgc_anchor_not_from_shared_ratio": float(rgc_not_shared / rgc_valid.sum()) if rgc_valid.any() else 0.0,
            "gdc_projected_shared_point_count": int(expected_valid.sum()),
            "gdc_anchor_valid_count": int(gdc_valid.sum()),
            "gdc_anchor_not_from_shared_count": gdc_not_shared,
            "gdc_anchor_not_from_shared_ratio": float(gdc_not_shared / gdc_valid.sum()) if gdc_valid.any() else 0.0,
        })
    output = Path(args.output_csv or root / "metrics" / "shared_anchor_protocol_audit.csv")
    summary = Path(args.summary_csv or root / "metrics" / "shared_anchor_protocol_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame_id"])
        writer.writeheader(); writer.writerows(rows)
    rgc_bad = sum(row["rgc_anchor_not_from_shared_count"] for row in rows)
    gdc_bad = sum(row["gdc_anchor_not_from_shared_count"] for row in rows)
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"]); writer.writeheader()
        writer.writerows([
            {"metric": "frame_count", "value": len(rows)},
            {"metric": "rgc_anchor_not_from_shared_count", "value": rgc_bad},
            {"metric": "rgc_anchor_not_from_shared_ratio", "value": 0.0 if rgc_bad == 0 else np.nan},
            {"metric": "gdc_anchor_not_from_shared_count", "value": gdc_bad},
            {"metric": "gdc_anchor_not_from_shared_ratio", "value": 0.0 if gdc_bad == 0 else np.nan},
        ])
    if rgc_bad or gdc_bad:
        raise SystemExit("Shared-anchor audit failed: a downstream anchor used an external point")
    print(f"Saved shared-anchor audit: {output}")


if __name__ == "__main__":
    main()
