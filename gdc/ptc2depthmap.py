import os
import argparse
import json
import hashlib
import sys
from pathlib import Path
import os.path as osp
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GDC_ROOT = Path(__file__).resolve().parent
if str(GDC_ROOT) not in sys.path:
    sys.path.insert(0, str(GDC_ROOT))
from data_utils.kitti_util import Calibration, load_velo_scan, load_image
from data_utils.kitti_object import get_lidar_in_image_fov
from range_gdc.shared_canonical_anchor import shared_points_to_gdc_depth
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, Pool


def get_ptc_in_image(ptc, calib, img):
    img_height, img_width, _ = img.shape
    _, _, img_fov_inds = get_lidar_in_image_fov(
        ptc[:, :3], calib, 0, 0, img_width-1, img_height-1, True)
    ptc = ptc[img_fov_inds]

    return ptc

def get_depth_map(
    ptc, calib, img, collision_policy="legacy_last", source_indices=None,
    return_source_index=False,
):
    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) - 1
    ptc_image = calib.project_velo_to_image(ptc[:, :3])
    ptc_2d = np.round(ptc_image[:, :2]).astype(np.int32)
    depth_info = calib.project_velo_to_rect(ptc[:, :3])
    if collision_policy == "legacy_last":
        if return_source_index:
            raise ValueError("source-index output is only supported for nearest_positive")
        depth_map[ptc_2d[:, 1], ptc_2d[:, 0]] = depth_info[:, 2]
    elif collision_policy == "nearest_positive":
        depth_map = shared_points_to_gdc_depth(
            ptc, calib, img.shape, clip_distance=2.0, invalid_value=-1.0,
            source_indices=source_indices, return_source_index=return_source_index,
        )
    else:
        raise ValueError(f"Unknown collision policy: {collision_policy}")
    return depth_map


def build_parser():
    parser = argparse.ArgumentParser(description='gen depthmaps from pointclouds')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--calib_path', type=str, help='path to calibration files')
    parser.add_argument('--image_path', type=str, help='path to calibration images')
    parser.add_argument('--split_file', type=str, help='indices of scene to be corrected')
    parser.add_argument('--i', type=int, default=None)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--collision-policy', choices=('legacy_last', 'nearest_positive'), default='legacy_last')
    parser.add_argument('--provenance-json', default=None,
                        help='Optional metadata linking this image anchor to a shared source manifest.')
    parser.add_argument('--source-index-input-path', default=None,
                        help='Canonical spherical source-index directory.')
    parser.add_argument('--source-index-output-path', default=None,
                        help='Optional per-pixel winning original source-index directory.')
    parser.add_argument('--selected-rows', type=int, nargs='+', default=None)
    return parser


def canonical_source_ids(args, i, point_count):
    if args.source_index_input_path is None:
        return np.arange(point_count, dtype=np.int32)
    if not args.selected_rows:
        raise ValueError("--selected-rows is required with --source-index-input-path")
    grid = np.load(osp.join(args.source_index_input_path, f"{i:06d}.npy")).astype(np.int32)
    rows = np.asarray(args.selected_rows, dtype=np.int32)
    if grid.ndim != 2 or np.any(rows < 0) or np.any(rows >= grid.shape[0]):
        raise ValueError(f"{i:06d}: selected rows are outside canonical source grid")
    source_ids = np.unique(grid[rows][grid[rows] >= 0])
    source_ids.sort()
    if source_ids.size != point_count:
        raise ValueError(
            f"{i:06d}: shared point count {point_count} != canonical source count {source_ids.size}"
        )
    return source_ids.astype(np.int32)


def canonical_depth_and_source(args, i, ptc, calib, img):
    source_ids = canonical_source_ids(args, i, ptc.shape[0])
    return get_depth_map(
        ptc, calib, img, "nearest_positive", source_indices=source_ids,
        return_source_index=True,
    )


def convert_and_save(args, i):
    ptc = load_velo_scan(
                osp.join(args.input_path, "{:06d}.bin".format(i)))
    calib = Calibration(osp.join(args.calib_path,
                                    "{:06d}.txt".format(i)))
    img = load_image(osp.join(args.image_path, "{:06d}.png".format(i)))
    if args.collision_policy == "nearest_positive":
        depth_map, source_map = canonical_depth_and_source(args, i, ptc, calib, img)
        if args.source_index_output_path:
            np.save(osp.join(args.source_index_output_path, f"{i:06d}.npy"), source_map)
    else:
        depth_map = get_depth_map(get_ptc_in_image(ptc, calib, img), calib, img, args.collision_policy)
    np.save(osp.join(args.output_path, "{:06d}".format(i)), depth_map)

def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.source_index_output_path and not os.path.exists(args.source_index_output_path):
        os.makedirs(args.source_index_output_path)
    if args.i is not None:
        i = args.i
        ptc = load_velo_scan(osp.join(args.input_path, "{:06d}.bin".format(i)))
        calib = Calibration(osp.join(args.calib_path, "{:06d}.txt".format(i)))
        img = load_image(osp.join(args.image_path, "{:06d}.png".format(i)))
        if args.collision_policy == "nearest_positive":
            depth_map, source_map = canonical_depth_and_source(args, i, ptc, calib, img)
            if args.source_index_output_path:
                np.save(osp.join(args.source_index_output_path, f"{i:06d}.npy"), source_map)
        else:
            depth_map = get_depth_map(get_ptc_in_image(ptc, calib, img), calib, img, args.collision_policy)
        np.save(osp.join(args.output_path, "{:06d}".format(i)), depth_map)
    else:
        with open(args.split_file) as f:
            idx_list = [int(x.strip())
                for x in f.readlines() if len(x.strip()) > 0]
        pbar = tqdm(total=len(idx_list))
        def update(*a):
            pbar.update()

        pool = Pool(args.threads)
        res = []
        for i in idx_list:
            res.append((i, pool.apply_async(convert_and_save, args=(args, i),
                                            callback=update)))

        pool.close()
        pool.join()
        pbar.clear(nolock=False)
        pbar.close()
    if args.provenance_json:
        digest = hashlib.sha256()
        with open(args.provenance_json, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        payload = {
            'anchor_source': 'shared_canonical_pointcloud',
            'collision_policy': args.collision_policy,
            'clip_distance': 2.0,
            'image_fov_bounds': 'float_uv: 0<=u<width-1, 0<=v<height-1',
            'source_index_output_path': (
                None if not args.source_index_output_path
                else os.path.abspath(args.source_index_output_path)
            ),
            'source_provenance_path': os.path.abspath(args.provenance_json),
            'source_manifest_sha256': digest.hexdigest(),
        }
        with open(osp.join(args.output_path, 'anchor_provenance.json'), 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
