#!/usr/bin/env python3
"""Uniform 360-degree spherical KITTI depth/range projection utilities."""

import argparse
import csv
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
GDC_ROOT = REPO_ROOT / "gdc"
if str(GDC_ROOT) not in sys.path:
    sys.path.insert(0, str(GDC_ROOT))

from data_utils.kitti_util import Calibration, load_image, load_velo_scan  # noqa: E402


DEFAULT_RANGE_H = 64
DEFAULT_RANGE_W = 1024
DEFAULT_VMIN_DEG = -24.9
DEFAULT_VMAX_DEG = 2.0
DEFAULT_AZIMUTH_MODE = "full_360_front_centered"
DEFAULT_DEPTH_MIN = 0.1
DEFAULT_DEPTH_MAX = 80.0
DEFAULT_INVALID_VALUE = 0.0


def read_split_ids(split_file):
    with open(split_file) as f:
        return [int(x.strip()) for x in f.readlines() if x.strip()]


def discover_ids(input_path):
    ids = []
    for name in sorted(os.listdir(input_path)):
        stem, ext = osp.splitext(name)
        if ext == ".npy" and stem.isdigit():
            ids.append(int(stem))
    if not ids:
        raise RuntimeError(f"No numeric .npy inputs found in {input_path}")
    return ids


def load_kitti_calib(calib_path):
    return Calibration(calib_path)


def depth_to_velodyne_points(depth, calib, depth_min=DEFAULT_DEPTH_MIN, depth_max=DEFAULT_DEPTH_MAX):
    """Backproject camera z-depth to Velodyne points using KITTI P2/R0/Tr calibration."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > float(depth_min)) & (depth < float(depth_max))
    rows, cols = np.where(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[rows, cols].astype(np.float32)
    uv_depth = np.column_stack((cols.astype(np.float32), rows.astype(np.float32), z))
    pts_rect = calib.project_image_to_rect(uv_depth)
    pts_velo = calib.project_rect_to_velo(pts_rect)
    return pts_velo.astype(np.float32)


def make_uniform_vertical_grid(range_h=DEFAULT_RANGE_H, vmin_deg=DEFAULT_VMIN_DEG, vmax_deg=DEFAULT_VMAX_DEG):
    if int(range_h) <= 0:
        raise ValueError("range_h must be positive")
    if float(vmax_deg) <= float(vmin_deg):
        raise ValueError("vmax_deg must be larger than vmin_deg")
    edges_bottom_to_top = np.linspace(float(vmin_deg), float(vmax_deg), int(range_h) + 1)
    centers_bottom_to_top = 0.5 * (edges_bottom_to_top[:-1] + edges_bottom_to_top[1:])
    return edges_bottom_to_top[::-1], centers_bottom_to_top[::-1]


def normalize_azimuth_mode(azimuth_mode):
    if azimuth_mode == "front_center":
        warnings.warn(
            "azimuth_mode='front_center' is deprecated; use "
            "'full_360_front_centered'. The projection is unchanged.",
            FutureWarning,
            stacklevel=2,
        )
        return "full_360_front_centered"
    if azimuth_mode not in {"full_360_front_centered", "bounded"}:
        raise ValueError(f"Unsupported azimuth_mode: {azimuth_mode}")
    return azimuth_mode


def azimuth_bounds(azimuth_mode=DEFAULT_AZIMUTH_MODE, azimuth_min_deg=None, azimuth_max_deg=None):
    mode = normalize_azimuth_mode(azimuth_mode)
    if mode == "full_360_front_centered":
        return mode, -180.0, 180.0
    if azimuth_min_deg is None or azimuth_max_deg is None:
        raise ValueError("bounded azimuth_mode requires azimuth_min_deg and azimuth_max_deg")
    if float(azimuth_max_deg) <= float(azimuth_min_deg):
        raise ValueError("azimuth_max_deg must be larger than azimuth_min_deg")
    return mode, float(azimuth_min_deg), float(azimuth_max_deg)


def make_uniform_azimuth_grid(
    range_w=DEFAULT_RANGE_W,
    azimuth_mode=DEFAULT_AZIMUTH_MODE,
    azimuth_min_deg=None,
    azimuth_max_deg=None,
):
    if int(range_w) <= 0:
        raise ValueError("range_w must be positive")
    _, amin, amax = azimuth_bounds(azimuth_mode, azimuth_min_deg, azimuth_max_deg)
    edges = np.linspace(amin, amax, int(range_w) + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def elevation_to_row_uniform(elevation_deg, range_h=DEFAULT_RANGE_H, vmin_deg=DEFAULT_VMIN_DEG, vmax_deg=DEFAULT_VMAX_DEG):
    step = (float(vmax_deg) - float(vmin_deg)) / int(range_h)
    return np.floor((float(vmax_deg) - elevation_deg) / step).astype(np.int64)


def azimuth_to_col_uniform(
    azimuth_deg,
    range_w=DEFAULT_RANGE_W,
    azimuth_mode=DEFAULT_AZIMUTH_MODE,
    azimuth_min_deg=None,
    azimuth_max_deg=None,
):
    _, amin, amax = azimuth_bounds(azimuth_mode, azimuth_min_deg, azimuth_max_deg)
    step = (amax - amin) / int(range_w)
    return np.floor((azimuth_deg - amin) / step).astype(np.int64)


def lidar_points_to_spherical_guide_uniform(
    points,
    range_h=DEFAULT_RANGE_H,
    range_w=DEFAULT_RANGE_W,
    vmin_deg=DEFAULT_VMIN_DEG,
    vmax_deg=DEFAULT_VMAX_DEG,
    azimuth_mode=DEFAULT_AZIMUTH_MODE,
    azimuth_min_deg=None,
    azimuth_max_deg=None,
    range_min=DEFAULT_DEPTH_MIN,
    range_max=DEFAULT_DEPTH_MAX,
    invalid_value=DEFAULT_INVALID_VALUE,
    return_count=False,
):
    range_img = np.full((int(range_h), int(range_w)), np.inf, dtype=np.float32)
    count = np.zeros((int(range_h), int(range_w)), dtype=np.int32)
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        range_img[:] = float(invalid_value)
        mask = np.zeros_like(range_img, dtype=bool)
        return (range_img, mask, count) if return_count else (range_img, mask)

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
        range_img[:] = float(invalid_value)
        mask = np.zeros_like(range_img, dtype=bool)
        return (range_img, mask, count) if return_count else (range_img, mask)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    ranges = ranges[valid]
    horizontal = horizontal[valid]
    elevation_deg = np.degrees(np.arctan2(z, horizontal))
    azimuth_deg = np.degrees(np.arctan2(y, x))
    row = elevation_to_row_uniform(elevation_deg, range_h, vmin_deg, vmax_deg)
    col = azimuth_to_col_uniform(
        azimuth_deg, range_w, azimuth_mode, azimuth_min_deg, azimuth_max_deg
    )
    inside = (row >= 0) & (row < int(range_h)) & (col >= 0) & (col < int(range_w))
    if np.any(inside):
        row = row[inside]
        col = col[inside]
        ranges = ranges[inside].astype(np.float32)
        np.minimum.at(range_img, (row, col), ranges)
        np.add.at(count, (row, col), 1)

    mask = np.isfinite(range_img)
    range_img[~mask] = float(invalid_value)
    return (range_img.astype(np.float32), mask, count) if return_count else (range_img.astype(np.float32), mask)


def make_selected_rows(range_h=DEFAULT_RANGE_H, row_offset=0, row_stride=None, anchor_rows=None):
    if anchor_rows is not None:
        rows = np.asarray(anchor_rows, dtype=np.int32)
    elif row_stride is not None:
        rows = np.arange(int(row_offset), int(range_h), int(row_stride), dtype=np.int32)
    else:
        rows = np.arange(int(range_h), dtype=np.int32)
    if rows.size == 0:
        raise ValueError("selected rows must not be empty")
    if np.any(rows < 0) or np.any(rows >= int(range_h)):
        raise ValueError(f"selected rows contain values outside [0, {int(range_h)})")
    return np.unique(rows)


def make_lowres_from_highres_guide(g64_range, g64_mask, selected_rows):
    selected_rows = np.asarray(selected_rows, dtype=np.int32)
    return g64_range[selected_rows, :].copy(), g64_mask[selected_rows, :].copy()


def projection_metadata(
    range_h=DEFAULT_RANGE_H,
    range_w=DEFAULT_RANGE_W,
    vmin_deg=DEFAULT_VMIN_DEG,
    vmax_deg=DEFAULT_VMAX_DEG,
    azimuth_mode=DEFAULT_AZIMUTH_MODE,
    azimuth_min_deg=None,
    azimuth_max_deg=None,
    invalid_value=DEFAULT_INVALID_VALUE,
    selected_rows=None,
    camera_fov_only=False,
):
    vertical_edges_deg, vertical_centers_deg = make_uniform_vertical_grid(range_h, vmin_deg, vmax_deg)
    canonical_mode, amin, amax = azimuth_bounds(
        azimuth_mode, azimuth_min_deg, azimuth_max_deg
    )
    azimuth_edges_deg, azimuth_centers_deg = make_uniform_azimuth_grid(
        range_w, canonical_mode, amin, amax
    )
    if selected_rows is None:
        selected_rows = np.arange(int(range_h), dtype=np.int32)
    return {
        "range_h": np.asarray(range_h, dtype=np.int32),
        "range_w": np.asarray(range_w, dtype=np.int32),
        "height": np.asarray(range_h, dtype=np.int32),
        "width": np.asarray(range_w, dtype=np.int32),
        "vmin_deg": np.asarray(vmin_deg, dtype=np.float64),
        "vmax_deg": np.asarray(vmax_deg, dtype=np.float64),
        "azimuth_mode": np.asarray(canonical_mode),
        "azimuth_min_deg": np.asarray(amin, dtype=np.float64),
        "azimuth_max_deg": np.asarray(amax, dtype=np.float64),
        "azimuth_span_deg": np.asarray(amax - amin, dtype=np.float64),
        "horizontal_resolution_deg": np.asarray((amax - amin) / int(range_w), dtype=np.float64),
        "vertical_resolution_deg": np.asarray((float(vmax_deg) - float(vmin_deg)) / int(range_h), dtype=np.float64),
        "row0_convention": np.asarray("top_highest_elevation"),
        "row0_elevation_direction": np.asarray("highest_to_lowest"),
        "camera_fov_only": np.asarray(bool(camera_fov_only)),
        "valid_column_count": np.asarray(range_w, dtype=np.int32),
        "valid_azimuth_span_deg": np.asarray(amax - amin, dtype=np.float64),
        "invalid_value": np.asarray(invalid_value, dtype=np.float32),
        "selected_rows": np.asarray(selected_rows, dtype=np.int32),
        "vertical_edges_deg": vertical_edges_deg.astype(np.float64),
        "vertical_centers_deg": vertical_centers_deg.astype(np.float64),
        "azimuth_edges_deg": azimuth_edges_deg.astype(np.float64),
        "azimuth_centers_deg": azimuth_centers_deg.astype(np.float64),
    }


def save_projection_meta(meta_dir, scene_id, metadata, count=None):
    os.makedirs(meta_dir, exist_ok=True)
    stem = f"{int(scene_id):06d}"
    for name in (
        "selected_rows",
        "vertical_edges_deg",
        "vertical_centers_deg",
        "azimuth_edges_deg",
        "azimuth_centers_deg",
    ):
        np.save(osp.join(meta_dir, f"{stem}_{name}.npy"), metadata[name])
    payload = dict(metadata)
    if count is not None:
        payload["projection_count"] = count.astype(np.int32)
    np.savez(osp.join(meta_dir, f"{stem}_guide_projection_meta.npz"), **payload)


def save_global_projection_meta(path, metadata):
    if not path:
        return
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # The global file is geometry-only and may be reused by several sources.
    # Source occupancy/FOV fields belong to each frame's metadata and must not
    # be silently overwritten by a later projection stage.
    source_specific = {
        "camera_fov_only",
        "valid_column_count",
        "valid_azimuth_span_deg",
        "valid_azimuth_min_deg",
        "valid_azimuth_max_deg",
        "occupied_column_indices",
    }
    payload = {k: v for k, v in metadata.items() if k not in source_specific}
    payload["metadata_kind"] = np.asarray("projection_geometry")
    np.savez(path, **payload)


def output_dirs(outdir, selected_rows):
    g_low = f"G{len(selected_rows)}"
    return {
        "g64_range": osp.join(outdir, "G64_range"),
        "g64_mask": osp.join(outdir, "G64_mask"),
        "glow_range": osp.join(outdir, f"{g_low}_range"),
        "glow_mask": osp.join(outdir, f"{g_low}_mask"),
        "meta": osp.join(outdir, "meta"),
    }


def convert_depth_file(scene_id, args):
    depth = np.load(osp.join(args.depth_npy, f"{int(scene_id):06d}.npy"))
    calib = load_kitti_calib(osp.join(args.calib, f"{int(scene_id):06d}.txt"))
    points = depth_to_velodyne_points(depth, calib, args.depth_min, args.depth_max)
    guide, mask, count = lidar_points_to_spherical_guide_uniform(
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
        return_count=True,
    )
    return save_guide_outputs(scene_id, depth.shape, guide, mask, count, args)


def convert_points_file(scene_id, args):
    points = load_velo_scan(osp.join(args.depth_npy, f"{int(scene_id):06d}.bin"))[:, :3]
    if args.image_fov_only:
        calib = load_kitti_calib(osp.join(args.calib, f"{int(scene_id):06d}.txt"))
        image = load_image(osp.join(args.image, f"{int(scene_id):06d}.png"))
        pts_img = calib.project_velo_to_image(points)
        pts_rect = calib.project_velo_to_rect(points)
        h, w = image.shape[:2]
        inside = (
            np.isfinite(pts_rect[:, 2])
            & (pts_rect[:, 2] > 0)
            & (pts_img[:, 0] >= 0)
            & (pts_img[:, 0] < w)
            & (pts_img[:, 1] >= 0)
            & (pts_img[:, 1] < h)
        )
        points = points[inside]
    guide, mask, count = lidar_points_to_spherical_guide_uniform(
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
        return_count=True,
    )
    return save_guide_outputs(scene_id, None, guide, mask, count, args)


def save_guide_outputs(scene_id, depth_shape, guide, mask, count, args):
    selected_rows = make_selected_rows(args.range_h, args.row_offset, args.row_stride, args.anchor_rows)
    dirs = output_dirs(args.outdir, selected_rows)
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    stem = f"{int(scene_id):06d}"
    low_range, low_mask = make_lowres_from_highres_guide(guide, mask, selected_rows)
    np.save(osp.join(dirs["g64_range"], f"{stem}.npy"), guide.astype(np.float32))
    np.save(osp.join(dirs["g64_mask"], f"{stem}.npy"), mask.astype(bool))
    np.save(osp.join(dirs["glow_range"], f"{stem}.npy"), low_range.astype(np.float32))
    np.save(osp.join(dirs["glow_mask"], f"{stem}.npy"), low_mask.astype(bool))

    metadata = projection_metadata(
        args.range_h,
        args.range_w,
        args.vmin_deg,
        args.vmax_deg,
        args.azimuth_mode,
        args.azimuth_min_deg,
        args.azimuth_max_deg,
        args.invalid_value,
        selected_rows,
        getattr(args, "image_fov_only", False),
    )
    frame_metadata = dict(metadata)
    occupied_columns = np.where(mask.any(axis=0))[0].astype(np.int32)
    frame_metadata["occupied_column_indices"] = occupied_columns
    frame_metadata["valid_column_count"] = np.asarray(
        occupied_columns.size, dtype=np.int32
    )
    if occupied_columns.size:
        az_centers = metadata["azimuth_centers_deg"]
        resolution = float(metadata["horizontal_resolution_deg"])
        az_min = float(az_centers[occupied_columns[0]] - 0.5 * resolution)
        az_max = float(az_centers[occupied_columns[-1]] + 0.5 * resolution)
        frame_metadata["valid_azimuth_min_deg"] = np.asarray(az_min, dtype=np.float64)
        frame_metadata["valid_azimuth_max_deg"] = np.asarray(az_max, dtype=np.float64)
        frame_metadata["valid_azimuth_span_deg"] = np.asarray(
            az_max - az_min, dtype=np.float64
        )
    else:
        frame_metadata["valid_azimuth_min_deg"] = np.asarray(np.nan, dtype=np.float64)
        frame_metadata["valid_azimuth_max_deg"] = np.asarray(np.nan, dtype=np.float64)
        frame_metadata["valid_azimuth_span_deg"] = np.asarray(0.0, dtype=np.float64)
    frame_metadata["metadata_kind"] = np.asarray("projection_source_frame")
    if getattr(args, "input_kind", "depth") == "points":
        occupied_rows = np.where(mask.any(axis=1))[0].astype(np.int32)
        frame_metadata["selected_rows"] = occupied_rows
    save_projection_meta(dirs["meta"], scene_id, frame_metadata, count=count)

    valid_ranges = guide[mask]
    return {
        "frame_id": stem,
        "depth_shape": "" if depth_shape is None else "x".join(str(v) for v in depth_shape),
        "image_shape": "",
        "G64_shape": "x".join(str(v) for v in guide.shape),
        "G64_valid_pixels": int(mask.sum()),
        "G64_valid_ratio": float(mask.mean()),
        "G64_min_range": float(np.min(valid_ranges)) if valid_ranges.size else np.nan,
        "G64_median_range": float(np.median(valid_ranges)) if valid_ranges.size else np.nan,
        "G64_max_range": float(np.max(valid_ranges)) if valid_ranges.size else np.nan,
        "selected_rows": " ".join(str(int(v)) for v in selected_rows),
        "G_low_shape": "x".join(str(v) for v in low_range.shape),
        "G_low_valid_pixels": int(low_mask.sum()),
        "projection_collision_count_sum": int(np.sum(np.maximum(count - 1, 0))),
        "projection_collision_pixel_count": int(np.sum(count > 1)),
    }


def write_stats_csv(path, rows):
    if not path:
        return
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "frame_id",
        "depth_shape",
        "image_shape",
        "G64_shape",
        "G64_valid_pixels",
        "G64_valid_ratio",
        "G64_min_range",
        "G64_median_range",
        "G64_max_range",
        "selected_rows",
        "G_low_shape",
        "G_low_valid_pixels",
        "projection_collision_count_sum",
        "projection_collision_pixel_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _depth_task(task):
    scene_id, args = task
    return convert_depth_file(scene_id, args)


def _points_task(task):
    scene_id, args = task
    return convert_points_file(scene_id, args)


def run_batch(args, worker):
    ids = read_split_ids(args.split_file) if args.split_file else discover_ids(args.depth_npy)
    selected_rows = make_selected_rows(args.range_h, args.row_offset, args.row_stride, args.anchor_rows)
    metadata = projection_metadata(
        args.range_h,
        args.range_w,
        args.vmin_deg,
        args.vmax_deg,
        args.azimuth_mode,
        args.azimuth_min_deg,
        args.azimuth_max_deg,
        args.invalid_value,
        selected_rows,
        getattr(args, "image_fov_only", False),
    )
    save_global_projection_meta(args.meta_path, metadata)
    tasks = [(scene_id, args) for scene_id in ids]
    rows = []
    if args.workers <= 1:
        for task in tqdm(tasks):
            rows.append(worker(task))
    else:
        with Pool(args.workers) as pool:
            for row in tqdm(pool.imap_unordered(worker, tasks, chunksize=1), total=len(tasks)):
                rows.append(row)
    rows.sort(key=lambda row: row["frame_id"])
    write_stats_csv(args.stats_csv, rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-npy", "--depth_npy", dest="depth_npy", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split-file", "--split_file", dest="split_file", default=None)
    parser.add_argument("--range-h", "--range_h", dest="range_h", type=int, default=DEFAULT_RANGE_H)
    parser.add_argument("--range-w", "--range_w", dest="range_w", type=int, default=DEFAULT_RANGE_W)
    parser.add_argument("--vmin-deg", "--vmin_deg", dest="vmin_deg", type=float, default=DEFAULT_VMIN_DEG)
    parser.add_argument("--vmax-deg", "--vmax_deg", dest="vmax_deg", type=float, default=DEFAULT_VMAX_DEG)
    parser.add_argument("--azimuth-mode", "--azimuth_mode", dest="azimuth_mode", default=DEFAULT_AZIMUTH_MODE)
    parser.add_argument("--azimuth-min-deg", "--azimuth_min_deg", dest="azimuth_min_deg", type=float, default=None)
    parser.add_argument("--azimuth-max-deg", "--azimuth_max_deg", dest="azimuth_max_deg", type=float, default=None)
    parser.add_argument("--depth-min", "--depth_min", dest="depth_min", type=float, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth-max", "--depth_max", dest="depth_max", type=float, default=DEFAULT_DEPTH_MAX)
    parser.add_argument("--invalid-value", "--invalid_value", dest="invalid_value", type=float, default=DEFAULT_INVALID_VALUE)
    parser.add_argument("--row-offset", "--row_offset", dest="row_offset", type=int, default=0)
    parser.add_argument("--row-stride", "--row_stride", dest="row_stride", type=int, default=None)
    parser.add_argument("--anchor-rows", "--anchor_rows", dest="anchor_rows", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--meta-path", "--meta_path", dest="meta_path", default=None)
    parser.add_argument("--stats-csv", "--stats_csv", dest="stats_csv", default=None)
    parser.add_argument("--input-kind", choices=["depth", "points"], default="depth")
    parser.add_argument("--image-fov-only", "--image_fov_only", dest="image_fov_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.input_kind == "depth":
        run_batch(args, _depth_task)
    else:
        if args.image is None and args.image_fov_only:
            raise ValueError("--image is required when --image-fov-only is enabled")
        run_batch(args, _points_task)


if __name__ == "__main__":
    main()
