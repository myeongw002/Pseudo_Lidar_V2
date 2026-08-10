#!/usr/bin/env python3
"""Compatibility wrapper for uniform 360-degree spherical range projection."""

import argparse
import os
import os.path as osp
import re
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
GDC_ROOT = REPO_ROOT / "gdc"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GDC_ROOT) not in sys.path:
    sys.path.insert(0, str(GDC_ROOT))

from data_utils.kitti_util import Calibration, load_image  # noqa: E402
from src.pseudo_lidar.depth_to_range_uniform import (  # noqa: E402
    DEFAULT_AZIMUTH_MODE,
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_INVALID_VALUE,
    DEFAULT_RANGE_H,
    DEFAULT_RANGE_W,
    DEFAULT_VMAX_DEG,
    DEFAULT_VMIN_DEG,
    _depth_task,
    _points_task,
    lidar_points_to_spherical_guide_uniform,
    load_kitti_calib,
    make_uniform_azimuth_grid,
    make_uniform_vertical_grid,
    normalize_azimuth_mode,
    read_split_ids,
    run_batch as run_uniform_batch,
)


def normalize_scene_id(path):
    name = osp.splitext(osp.basename(path))[0]
    patterns = [
        r"_G\d+_corr_range$",
        r"_G\d+_corr_mask$",
        r"_G\d+_range$",
        r"_G\d+_mask$",
        r"_R\d+_range$",
        r"_R\d+_mask$",
        r"_range$",
        r"_mask$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_name = re.sub(pattern, "", name)
            if new_name != name:
                name = new_name
                changed = True
    return name


def find_input_npy(input_path, scene_id):
    exact = osp.join(input_path, f"{int(scene_id):06d}.npy")
    if osp.exists(exact):
        return exact
    matches = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if not name.endswith(".npy"):
                continue
            path = osp.join(root, name)
            if normalize_scene_id(path) == f"{int(scene_id):06d}":
                matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .npy input for {int(scene_id):06d} in {input_path}")
    raise RuntimeError(f"Multiple .npy inputs for {int(scene_id):06d}: {matches}")


def depth_to_velo(depth, calib, depth_min=DEFAULT_DEPTH_MIN, depth_max=DEFAULT_DEPTH_MAX):
    from src.pseudo_lidar.depth_to_range_uniform import depth_to_velodyne_points

    return depth_to_velodyne_points(depth, calib, depth_min, depth_max)


def points_to_range_image(points, args):
    range_img, _ = lidar_points_to_spherical_guide_uniform(
        points,
        range_h=args.range_h,
        range_w=args.range_w,
        vmin_deg=args.vmin_deg,
        vmax_deg=args.vmax_deg,
        azimuth_mode=args.azimuth_mode,
        azimuth_min_deg=args.azimuth_min_deg,
        azimuth_max_deg=args.azimuth_max_deg,
        range_min=args.depth_min,
        range_max=args.depth_max,
        invalid_value=args.invalid_value,
    )
    return range_img


def projection_centers_from_meta_or_args(args):
    if args.meta_path and osp.exists(args.meta_path):
        meta = np.load(args.meta_path, allow_pickle=True)
        if "vertical_centers_deg" in meta.files and "azimuth_centers_deg" in meta.files:
            return meta["vertical_centers_deg"], meta["azimuth_centers_deg"]
    _, vertical = make_uniform_vertical_grid(args.range_h, args.vmin_deg, args.vmax_deg)
    _, azimuth = make_uniform_azimuth_grid(
        args.range_w,
        args.azimuth_mode,
        args.azimuth_min_deg,
        args.azimuth_max_deg,
    )
    return vertical, azimuth


def range_to_velo(range_img, args):
    range_img = np.asarray(range_img, dtype=np.float32)
    vertical_centers_deg, azimuth_centers_deg = projection_centers_from_meta_or_args(args)
    expected_shape = (len(vertical_centers_deg), len(azimuth_centers_deg))
    if range_img.shape != expected_shape:
        raise ValueError(f"range image shape {range_img.shape} != projection metadata {expected_shape}")
    valid = np.isfinite(range_img) & (range_img > 0)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    ranges = range_img[rows, cols].astype(np.float64)
    elevation = np.deg2rad(vertical_centers_deg[rows])
    azimuth = np.deg2rad(azimuth_centers_deg[cols])

    horizontal = ranges * np.cos(elevation)
    x = horizontal * np.cos(azimuth)
    y = horizontal * np.sin(azimuth)
    z = ranges * np.sin(elevation)
    return np.column_stack((x, y, z)).astype(np.float32)


def velo_to_depth(points, calib, image_shape, invalid_value):
    height, width = image_shape[:2]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if points.size == 0:
        depth[:] = invalid_value
        return depth

    pts_img = calib.project_velo_to_image(points[:, :3])
    pts_rect = calib.project_velo_to_rect(points[:, :3])
    cols = np.round(pts_img[:, 0]).astype(np.int64)
    rows = np.round(pts_img[:, 1]).astype(np.int64)
    z = pts_rect[:, 2].astype(np.float32)
    inside = (
        np.isfinite(z)
        & (z > 0)
        & (cols >= 0)
        & (cols < width)
        & (rows >= 0)
        & (rows < height)
    )
    if np.any(inside):
        np.minimum.at(depth, (rows[inside], cols[inside]), z[inside])
    depth[~np.isfinite(depth)] = invalid_value
    return depth


def convert_range_to_depth(task):
    scene_id, args = task
    range_img = np.load(find_input_npy(args.input_path, scene_id))
    calib = Calibration(osp.join(args.calib_path, f"{int(scene_id):06d}.txt"))
    image = load_image(osp.join(args.image_path, f"{int(scene_id):06d}.png"))
    points = range_to_velo(range_img, args)
    depth = velo_to_depth(points, calib, image.shape, args.invalid_value)
    os.makedirs(args.output_path, exist_ok=True)
    np.save(osp.join(args.output_path, f"{int(scene_id):06d}.npy"), depth.astype(np.float32))


def run_depth_to_range(args):
    args.depth_npy = args.input_path
    args.calib = args.calib_path
    args.image = args.image_path
    args.outdir = args.output_path
    args.workers = args.threads
    run_uniform_batch(args, _depth_task)


def run_ptc_to_range(args):
    args.depth_npy = args.input_path
    args.calib = args.calib_path
    args.image = args.image_path
    args.outdir = args.output_path
    args.workers = args.threads
    args.input_kind = "points"
    run_uniform_batch(args, _points_task)


def run_range_to_depth(args):
    ids = read_split_ids(args.split_file)
    tasks = [(scene_id, args) for scene_id in ids]
    for task in tqdm(tasks):
        convert_range_to_depth(task)


def add_common_args(parser):
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--calib_path", required=True)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--height", "--range_h", dest="range_h", type=int, default=DEFAULT_RANGE_H)
    parser.add_argument("--width", "--range_w", dest="range_w", type=int, default=DEFAULT_RANGE_W)
    parser.add_argument("--vmin_deg", type=float, default=DEFAULT_VMIN_DEG)
    parser.add_argument("--vmax_deg", type=float, default=DEFAULT_VMAX_DEG)
    parser.add_argument("--azimuth_mode", default=DEFAULT_AZIMUTH_MODE)
    parser.add_argument("--azimuth_min_deg", type=float, default=None)
    parser.add_argument("--azimuth_max_deg", type=float, default=None)
    parser.add_argument("--depth_min", type=float, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=DEFAULT_DEPTH_MAX)
    parser.add_argument("--invalid_value", type=float, default=DEFAULT_INVALID_VALUE)
    parser.add_argument("--row_offset", type=int, default=0)
    parser.add_argument("--row_stride", type=int, default=None)
    parser.add_argument("--anchor_rows", type=int, nargs="+", default=None)
    parser.add_argument("--meta_path", default=None)
    parser.add_argument("--stats_csv", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--elevation_top_deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--vertical_step_deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--azimuth_left_deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--azimuth_fov_deg", type=float, default=None, help=argparse.SUPPRESS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    depth_to_range = subparsers.add_parser("depth-to-range")
    add_common_args(depth_to_range)

    ptc_to_range = subparsers.add_parser("ptc-to-range")
    add_common_args(ptc_to_range)
    ptc_to_range.add_argument("--image_fov_only", action=argparse.BooleanOptionalAction, default=True)

    range_to_depth = subparsers.add_parser("range-to-depth")
    add_common_args(range_to_depth)

    return parser.parse_args()


def main():
    args = parse_args()
    args.azimuth_mode = normalize_azimuth_mode(args.azimuth_mode)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.range_h != DEFAULT_RANGE_H or args.range_w != DEFAULT_RANGE_W:
        print(f"WARNING: non-default range shape requested: {(args.range_h, args.range_w)}")
    if args.mode == "depth-to-range":
        run_depth_to_range(args)
    elif args.mode == "ptc-to-range":
        if args.image_path is None and args.image_fov_only:
            raise ValueError("--image_path is required for ptc-to-range with --image_fov_only")
        run_ptc_to_range(args)
    elif args.mode == "range-to-depth":
        if args.image_path is None:
            raise ValueError("--image_path is required for range-to-depth")
        run_range_to_depth(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
