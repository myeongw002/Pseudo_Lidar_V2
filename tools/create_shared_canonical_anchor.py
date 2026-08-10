#!/usr/bin/env python3
"""Generate one canonical sparse original-LiDAR point set per KITTI frame."""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.shared_canonical_anchor import (
    extract_shared_points, project_with_source_indices, projection_definition,
    sha256_file, write_json,
)


def ids_from_split(path):
    with open(path) as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velodyne-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-pointcloud-dir", required=True)
    parser.add_argument("--output-source-index-dir", required=True)
    parser.add_argument("--provenance-json", required=True)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--vmin-deg", type=float, default=-24.9)
    parser.add_argument("--vmax-deg", type=float, default=2.0)
    parser.add_argument("--azimuth-mode", default="full_360_front_centered")
    parser.add_argument("--azimuth-min-deg", type=float, default=None)
    parser.add_argument("--azimuth-max-deg", type=float, default=None)
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=80.0)
    parser.add_argument("--invalid-value", type=float, default=0.0)
    parser.add_argument("--selected-rows", type=int, nargs="+", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    params = dict(range_h=args.height, range_w=args.width, vmin_deg=args.vmin_deg,
                  vmax_deg=args.vmax_deg, azimuth_mode=args.azimuth_mode,
                  azimuth_min_deg=args.azimuth_min_deg, azimuth_max_deg=args.azimuth_max_deg,
                  range_min=args.range_min, range_max=args.range_max,
                  invalid_value=args.invalid_value)
    selected_rows = np.unique(np.asarray(args.selected_rows, dtype=np.int32))
    if not selected_rows.size or np.any(selected_rows < 0) or np.any(selected_rows >= args.height):
        raise ValueError("--selected-rows must be unique rows inside the projection height")
    velo_dir = Path(args.velodyne_dir).resolve()
    pc_dir, index_dir = Path(args.output_pointcloud_dir), Path(args.output_source_index_dir)
    pc_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for scene_id in tqdm(ids_from_split(args.split_file)):
        stem = f"{scene_id:06d}"
        source_path = velo_dir / f"{stem}.bin"
        points = np.fromfile(source_path, dtype=np.float32)
        if points.size % 4:
            raise ValueError(f"{source_path}: invalid KITTI float32 x/y/z/intensity records")
        points = points.reshape(-1, 4)
        ranges, source_index, _ = project_with_source_indices(points, **params)
        shared, selected_indices = extract_shared_points(points, source_index, selected_rows)
        pc_path, index_path = pc_dir / f"{stem}.bin", index_dir / f"{stem}.npy"
        shared.astype(np.float32).tofile(pc_path)
        np.save(index_path, source_index.astype(np.int32))
        selected_valid = source_index[selected_rows] >= 0
        frames.append({
            "frame_id": stem, "source_velodyne_path": str(source_path),
            "input_point_count": int(points.shape[0]),
            "canonical_valid_cell_count": int((source_index >= 0).sum()),
            "selected_row_valid_cell_count": int(selected_valid.sum()),
            "shared_sparse_point_count": int(shared.shape[0]),
            "shared_sparse_output_path": str(pc_path),
            "source_index_output_path": str(index_path), "sha256": sha256_file(pc_path),
        })
    definition = projection_definition(**params)
    provenance = {
        "schema_version": 1, "source_velodyne_dir": str(velo_dir),
        "projection_definition": definition, **definition,
        "selected_rows": [int(v) for v in selected_rows], "frame_count": len(frames),
        "frames": frames,
    }
    write_json(args.provenance_json, provenance)
    print(f"Wrote canonical shared-anchor provenance: {args.provenance_json}")


if __name__ == "__main__":
    main()
