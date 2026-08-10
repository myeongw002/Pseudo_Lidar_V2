#!/usr/bin/env python3
"""Create a row-masked Range-GDC anchor from GT G64 range images."""

import argparse
import json
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

try:
    from .range_projection import find_input_npy
except ImportError:
    from range_projection import find_input_npy


def read_split_ids(split_file):
    with open(split_file) as f:
        return [int(x.strip()) for x in f.readlines() if x.strip()]


def selected_rows_array(values, height):
    rows = np.asarray(values, dtype=np.int32)
    if rows.size == 0:
        raise ValueError("--selected_rows must not be empty")
    if np.any(rows < 0) or np.any(rows >= int(height)):
        raise ValueError(f"--selected_rows contain rows outside [0, {height})")
    return np.unique(rows)


def default_meta_dir(output_range_path):
    path = osp.normpath(output_range_path)
    if osp.basename(path) == "G64_range":
        return osp.join(osp.dirname(path), "meta")
    return osp.join(path, "meta")


def copy_projection_meta(src, dst_dir, selected_rows):
    if not src or not osp.exists(src):
        return
    os.makedirs(dst_dir, exist_ok=True)
    meta = np.load(src, allow_pickle=True)
    payload = {key: meta[key] for key in meta.files}
    payload["selected_rows"] = selected_rows.astype(np.int32)
    np.savez(osp.join(dst_dir, "projection_meta.npz"), **payload)




def write_anchor_definition(args, selected_rows):
    os.makedirs(args.meta_dir, exist_ok=True)
    payload = {
        "mode": "gt_row_mask",
        "selected_rows": [int(v) for v in selected_rows.tolist()],
        "expected_height": int(args.expected_height),
        "expected_width": int(args.expected_width),
        "invalid_value": float(args.invalid_value),
        "source_gt_range_path": osp.abspath(args.gt_range_path),
        "split_file": osp.abspath(args.split_file),
        "projection_meta_path": (
            None
            if not args.projection_meta_path
            else osp.abspath(args.projection_meta_path)
        ),
    }
    with open(osp.join(args.meta_dir, "anchor_definition.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def process_one(scene_id, args):
    gt = np.load(find_input_npy(args.gt_range_path, scene_id)).astype(np.float32)
    expected_shape = (int(args.expected_height), int(args.expected_width))
    if gt.shape != expected_shape:
        raise ValueError(f"{scene_id:06d}: expected GT range shape {expected_shape}, got {gt.shape}")

    rows = selected_rows_array(args.selected_rows, gt.shape[0])
    valid = np.isfinite(gt) & (gt > float(args.invalid_value))
    anchor = np.full_like(gt, float(args.invalid_value), dtype=np.float32)
    mask = np.zeros_like(valid, dtype=bool)
    anchor[rows, :] = gt[rows, :]
    mask[rows, :] = valid[rows, :]
    anchor[~mask] = float(args.invalid_value)

    os.makedirs(args.output_range_path, exist_ok=True)
    os.makedirs(args.output_mask_path, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)
    stem = f"{scene_id:06d}"
    np.save(osp.join(args.output_range_path, f"{stem}.npy"), anchor.astype(np.float32))
    np.save(osp.join(args.output_mask_path, f"{stem}.npy"), mask.astype(bool))
    np.save(osp.join(args.meta_dir, f"{stem}_selected_rows.npy"), rows.astype(np.int32))

    actual_rows = np.where(mask.any(axis=1))[0]
    if not np.all(np.isin(actual_rows, rows)):
        raise ValueError(
            f"{scene_id:06d}: anchor contains rows outside configured rows: "
            f"configured={rows}, actual={actual_rows}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_range_path", required=True)
    parser.add_argument("--output_range_path", required=True)
    parser.add_argument("--output_mask_path", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--selected_rows", type=int, nargs="+", required=True)
    parser.add_argument("--invalid_value", type=float, default=0.0)
    parser.add_argument("--expected_height", type=int, default=64)
    parser.add_argument("--expected_width", type=int, default=1024)
    parser.add_argument("--projection_meta_path", default=None)
    parser.add_argument("--meta_dir", default=None)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.meta_dir is None:
        args.meta_dir = default_meta_dir(args.output_range_path)
    ids = read_split_ids(args.split_file)
    rows = selected_rows_array(args.selected_rows, args.expected_height)
    copy_projection_meta(args.projection_meta_path, args.meta_dir, rows)
    write_anchor_definition(args, rows)
    tasks = [(scene_id, args) for scene_id in ids]
    if args.threads <= 1:
        for task in tqdm(tasks):
            process_one(*task)
    else:
        with Pool(args.threads) as pool:
            list(tqdm(pool.starmap(process_one, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
