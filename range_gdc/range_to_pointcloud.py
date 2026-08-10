#!/usr/bin/env python3
"""Export Velodyne point clouds from uniform spherical range images."""

import argparse
import os
import os.path as osp
import re

import numpy as np

try:
    from src.pseudo_lidar.depth_to_range_uniform import (
        make_uniform_azimuth_grid,
        make_uniform_vertical_grid,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.pseudo_lidar.depth_to_range_uniform import (
        make_uniform_azimuth_grid,
        make_uniform_vertical_grid,
    )


def read_split_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def normalize_scene_id(path):
    name = osp.splitext(osp.basename(path))[0]
    for pattern in (r"_G\d+_corr_range$", r"_G\d+_corr_mask$", r"_range$", r"_mask$"):
        name = re.sub(pattern, "", name)
    return name


def find_input_npy(input_path, scene_id):
    exact = osp.join(input_path, f"{int(scene_id):06d}.npy")
    if osp.exists(exact):
        return exact
    matches = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.endswith(".npy") and normalize_scene_id(osp.join(root, name)) == f"{int(scene_id):06d}":
                matches.append(osp.join(root, name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .npy for {int(scene_id):06d} in {input_path}")
    raise RuntimeError(f"Multiple .npy files for {int(scene_id):06d}: {matches}")


def load_centers(meta_path, height, width, vmin_deg, vmax_deg, azimuth_mode, azimuth_min_deg, azimuth_max_deg):
    if meta_path and osp.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        if "vertical_centers_deg" in meta.files and "azimuth_centers_deg" in meta.files:
            return meta["vertical_centers_deg"], meta["azimuth_centers_deg"]
    _, vertical_centers = make_uniform_vertical_grid(height, vmin_deg, vmax_deg)
    _, azimuth_centers = make_uniform_azimuth_grid(
        width, azimuth_mode, azimuth_min_deg, azimuth_max_deg
    )
    return vertical_centers, azimuth_centers


def range_to_points(range_img, vertical_centers_deg, azimuth_centers_deg, range_min, range_max):
    valid = np.isfinite(range_img) & (range_img >= range_min) & (range_img <= range_max)
    rows, cols = np.where(valid)
    ranges = range_img[rows, cols].astype(np.float64)
    elevation = np.deg2rad(vertical_centers_deg[rows])
    azimuth = np.deg2rad(azimuth_centers_deg[cols])
    horizontal = ranges * np.cos(elevation)
    x = horizontal * np.cos(azimuth)
    y = horizontal * np.sin(azimuth)
    z = ranges * np.sin(elevation)
    intensity = np.zeros_like(x)
    return np.column_stack((x, y, z, intensity)).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--projection_meta_path", default=None)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--vmin_deg", type=float, default=-24.9)
    parser.add_argument("--vmax_deg", type=float, default=2.0)
    parser.add_argument("--azimuth_mode", default="full_360_front_centered")
    parser.add_argument("--azimuth_min_deg", type=float, default=None)
    parser.add_argument("--azimuth_max_deg", type=float, default=None)
    parser.add_argument("--range_min", type=float, default=0.1)
    parser.add_argument("--range_max", type=float, default=80.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ids = read_split_ids(args.split_file)
    vertical_centers, azimuth_centers = load_centers(
        args.projection_meta_path,
        args.height,
        args.width,
        args.vmin_deg,
        args.vmax_deg,
        args.azimuth_mode,
        args.azimuth_min_deg,
        args.azimuth_max_deg,
    )
    os.makedirs(args.output_path, exist_ok=True)
    expected_shape = (len(vertical_centers), len(azimuth_centers))
    for scene_id in ids:
        range_img = np.load(find_input_npy(args.range_path, scene_id)).astype(np.float32)
        if range_img.shape != expected_shape:
            raise ValueError(f"{scene_id:06d}: expected {expected_shape}, got {range_img.shape}")
        points = range_to_points(
            range_img,
            vertical_centers,
            azimuth_centers,
            args.range_min,
            args.range_max,
        )
        if points.shape[0] <= 0:
            raise ValueError(f"{scene_id:06d}: exported point cloud would be empty")
        points.tofile(osp.join(args.output_path, f"{int(scene_id):06d}.bin"))
    print(f"Saved {len(ids)} point cloud(s) to {args.output_path}")


if __name__ == "__main__":
    main()
