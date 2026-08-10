#!/usr/bin/env python3
"""Create deterministic PNG previews from depth or range .npy files."""

import argparse
import os
import os.path as osp
import re

import numpy as np


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


def save_preview(path, array, vmin, vmax, cmap):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import cm
    from matplotlib import image as mpimg

    valid = np.isfinite(array) & (array > vmin) & (array < vmax)
    norm = np.clip((array.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.get_cmap(cmap)(norm, bytes=True)
    rgba[~valid] = np.array([0, 0, 0, 255], dtype=np.uint8)
    os.makedirs(osp.dirname(path), exist_ok=True)
    mpimg.imsave(path, rgba)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--vmin", type=float, default=0.1)
    parser.add_argument("--vmax", type=float, default=80.0)
    parser.add_argument("--cmap", default="turbo")
    return parser.parse_args()


def main():
    args = parse_args()
    ids = read_split_ids(args.split_file)
    if args.max_items is not None:
        ids = ids[: int(args.max_items)]
    for scene_id in ids:
        array = np.load(find_input_npy(args.input_path, scene_id))
        save_preview(
            osp.join(args.output_path, f"{int(scene_id):06d}.png"),
            array,
            args.vmin,
            args.vmax,
            args.cmap,
        )
    print(f"Saved {len(ids)} preview(s) to {args.output_path}")


if __name__ == "__main__":
    main()
