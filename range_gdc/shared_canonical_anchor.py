#!/usr/bin/env python3
"""Create and consume the canonical sparse LiDAR anchor.

The source-index grid is deliberately kept alongside the range grid: it makes
the collision winner observable and prevents downstream users from silently
reconstructing points from spherical cell centres.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pseudo_lidar.depth_to_range_uniform import (
    azimuth_to_col_uniform,
    elevation_to_row_uniform,
    azimuth_bounds,
)


INVALID_SOURCE_INDEX = -1


def project_with_source_indices(
    points, range_h, range_w, vmin_deg, vmax_deg, azimuth_mode,
    azimuth_min_deg=None, azimuth_max_deg=None, range_min=0.1,
    range_max=80.0, invalid_value=0.0,
):
    """Project original points and return range, source-index, and occupancy.

    The winner is the nearest spherical-range point; exact range ties are
    broken by the original PCD index, making output stable across platforms.
    """
    points = np.asarray(points, dtype=np.float32)
    shape = (int(range_h), int(range_w))
    ranges_img = np.full(shape, float(invalid_value), dtype=np.float32)
    source = np.full(shape, INVALID_SOURCE_INDEX, dtype=np.int32)
    count = np.zeros(shape, dtype=np.int32)
    if points.size == 0:
        return ranges_img, source, count
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")

    xyz = points[:, :3].astype(np.float64)
    original_index = np.arange(xyz.shape[0], dtype=np.int32)
    x, y, z = xyz.T
    horizontal = np.hypot(x, y)
    spherical_range = np.linalg.norm(xyz, axis=1)
    valid = (
        np.isfinite(spherical_range) & (spherical_range > float(range_min))
        & (spherical_range < float(range_max)) & (horizontal > 0.0)
    )
    if not np.any(valid):
        return ranges_img, source, count
    elevation = np.degrees(np.arctan2(z[valid], horizontal[valid]))
    azimuth = np.degrees(np.arctan2(y[valid], x[valid]))
    rows = elevation_to_row_uniform(elevation, range_h, vmin_deg, vmax_deg)
    cols = azimuth_to_col_uniform(
        azimuth, range_w, azimuth_mode, azimuth_min_deg, azimuth_max_deg
    )
    candidate_indices = original_index[valid]
    candidate_ranges = spherical_range[valid].astype(np.float32)
    inside = ((rows >= 0) & (rows < int(range_h)) & (cols >= 0)
              & (cols < int(range_w)))
    rows, cols = rows[inside], cols[inside]
    candidate_indices = candidate_indices[inside]
    candidate_ranges = candidate_ranges[inside]
    if not rows.size:
        return ranges_img, source, count
    np.add.at(count, (rows, cols), 1)
    flat = rows * int(range_w) + cols
    # lexsort's final key is primary: range first, original index second.
    winner_order = np.lexsort((candidate_indices, candidate_ranges, flat))
    sorted_flat = flat[winner_order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    winners = winner_order[first]
    ranges_img[rows[winners], cols[winners]] = candidate_ranges[winners]
    source[rows[winners], cols[winners]] = candidate_indices[winners]
    return ranges_img, source, count


def extract_shared_points(original_points, source_index, selected_rows):
    """Return exact original x/y/z/intensity records in source-index order."""
    original_points = np.asarray(original_points, dtype=np.float32)
    grid = np.asarray(source_index, dtype=np.int32)
    rows = np.asarray(selected_rows, dtype=np.int32)
    if grid.ndim != 2 or np.any(rows < 0) or np.any(rows >= grid.shape[0]):
        raise ValueError("selected_rows are outside source-index grid")
    indices = np.unique(grid[rows, :][grid[rows, :] >= 0])
    indices.sort()
    if np.any(indices >= original_points.shape[0]):
        raise ValueError("source index exceeds original point cloud size")
    return original_points[indices].copy(), indices.astype(np.int32)


def shared_points_to_camera_depth(
    points, calib, image_shape, invalid_value=-1.0, source_indices=None,
    return_source_index=False,
):
    """Project points with coupled nearest-positive depth/source winners."""
    height, width = image_shape[:2]
    depth = np.full((height, width), float(invalid_value), dtype=np.float32)
    source_map = np.full((height, width), INVALID_SOURCE_INDEX, dtype=np.int32)
    points = np.asarray(points, dtype=np.float32)
    if source_indices is None:
        source_indices = np.arange(points.shape[0] if points.ndim else 0, dtype=np.int32)
    source_indices = np.asarray(source_indices, dtype=np.int32)
    if points.size == 0:
        return (depth, source_map) if return_source_index else depth
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if source_indices.shape != (points.shape[0],):
        raise ValueError("source_indices must have one entry per point")
    image = calib.project_velo_to_image(points[:, :3])
    rect = calib.project_velo_to_rect(points[:, :3])
    cols = np.round(image[:, 0]).astype(np.int64)
    rows = np.round(image[:, 1]).astype(np.int64)
    z = rect[:, 2].astype(np.float32)
    valid = (np.isfinite(z) & (z > 0) & (cols >= 0) & (cols < width)
             & (rows >= 0) & (rows < height))
    if np.any(valid):
        valid_rows, valid_cols, valid_z = rows[valid], cols[valid], z[valid]
        valid_source = source_indices[valid]
        flat = valid_rows * width + valid_cols
        order = np.lexsort((valid_source, valid_z, flat))
        ordered_flat = flat[order]
        winners = order[np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]]
        depth[valid_rows[winners], valid_cols[winners]] = valid_z[winners]
        source_map[valid_rows[winners], valid_cols[winners]] = valid_source[winners]
    return (depth, source_map) if return_source_index else depth


def shared_points_to_gdc_depth(
    points, calib, image_shape, clip_distance=2.0, invalid_value=-1.0,
    source_indices=None, return_source_index=False,
):
    """Apply canonical GDC FOV filtering and nearest-positive camera-z projection.

    This intentionally reproduces ``get_ptc_in_image`` semantics: FOV inclusion
    uses floating projected coordinates with exclusive ``width - 1`` and
    ``height - 1`` upper bounds, and points require Velodyne ``x`` strictly
    greater than ``clip_distance``. Pixel assignment remains rounded and uses
    the nearest positive rect-camera z for collisions.
    """
    points = np.asarray(points, dtype=np.float32)
    if source_indices is None:
        source_indices = np.arange(points.shape[0] if points.ndim else 0, dtype=np.int32)
    source_indices = np.asarray(source_indices, dtype=np.int32)
    if points.size == 0:
        return shared_points_to_camera_depth(
            points, calib, image_shape, invalid_value=invalid_value,
            source_indices=source_indices, return_source_index=return_source_index,
        )
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if source_indices.shape != (points.shape[0],):
        raise ValueError("source_indices must have one entry per point")
    height, width = image_shape[:2]
    projected = calib.project_velo_to_image(points[:, :3])
    fov = (
        (projected[:, 0] < width - 1)
        & (projected[:, 0] >= 0)
        & (projected[:, 1] < height - 1)
        & (projected[:, 1] >= 0)
        & (points[:, 0] > float(clip_distance))
    )
    return shared_points_to_camera_depth(
        points[fov], calib, image_shape, invalid_value=invalid_value,
        source_indices=source_indices[fov], return_source_index=return_source_index,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_definition(**kwargs):
    mode, amin, amax = azimuth_bounds(
        kwargs["azimuth_mode"], kwargs.get("azimuth_min_deg"), kwargs.get("azimuth_max_deg")
    )
    return {
        "height": int(kwargs["range_h"]), "width": int(kwargs["range_w"]),
        "vmin_deg": float(kwargs["vmin_deg"]), "vmax_deg": float(kwargs["vmax_deg"]),
        "azimuth_mode": mode, "azimuth_min_deg": float(amin),
        "azimuth_max_deg": float(amax), "range_min": float(kwargs["range_min"]),
        "range_max": float(kwargs["range_max"]),
        "invalid_value": float(kwargs.get("invalid_value", 0.0)),
        "collision_policy": "nearest_spherical_range_then_lowest_source_index",
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
