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
from data_utils.kitti_util import Calibration, load_velo_scan, load_image
from data_utils.kitti_object import get_lidar_in_image_fov
from range_gdc.shared_canonical_anchor import shared_points_to_camera_depth
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, Pool


def get_ptc_in_image(ptc, calib, img):
    img_height, img_width, _ = img.shape
    _, _, img_fov_inds = get_lidar_in_image_fov(
        ptc[:, :3], calib, 0, 0, img_width-1, img_height-1, True)
    ptc = ptc[img_fov_inds]

    return ptc

def get_depth_map(ptc, calib, img, collision_policy="legacy_last"):
    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) - 1
    ptc_image = calib.project_velo_to_image(ptc[:, :3])
    ptc_2d = np.round(ptc_image[:, :2]).astype(np.int32)
    depth_info = calib.project_velo_to_rect(ptc[:, :3])
    positive = np.isfinite(depth_info[:, 2]) & (depth_info[:, 2] > 0)
    if collision_policy == "legacy_last":
        depth_map[ptc_2d[:, 1], ptc_2d[:, 0]] = depth_info[:, 2]
    elif collision_policy == "nearest_positive":
        depth_map = shared_points_to_camera_depth(ptc, calib, img.shape, invalid_value=-1.0)
    else:
        raise ValueError(f"Unknown collision policy: {collision_policy}")
    return depth_map


parser = argparse.ArgumentParser(description='gen depthmaps from pointclouds')
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--calib_path', type=str,
                    help='path to calibration files')
parser.add_argument('--image_path', type=str,
                    help='path to calibration images')
parser.add_argument('--split_file', type=str,
                    help='indices of scene to be corrected')
parser.add_argument('--i', type=int, default=None)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--collision-policy', choices=('legacy_last', 'nearest_positive'), default='legacy_last')
parser.add_argument('--provenance-json', default=None,
                    help='Optional metadata linking this image anchor to a shared source manifest.')
args = parser.parse_args()


def convert_and_save(args, i):
    ptc = load_velo_scan(
                osp.join(args.input_path, "{:06d}.bin".format(i)))
    calib = Calibration(osp.join(args.calib_path,
                                    "{:06d}.txt".format(i)))
    img = load_image(osp.join(args.image_path, "{:06d}.png".format(i)))
    ptc = get_ptc_in_image(ptc, calib, img)
    depth_map = get_depth_map(ptc, calib, img, args.collision_policy)
    np.save(osp.join(args.output_path, "{:06d}".format(i)), depth_map)

if __name__ == "__main__":
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.i is not None:
        i = args.i
        ptc = load_velo_scan(osp.join(args.input_path, "{:06d}.bin".format(i)))
        calib = Calibration(osp.join(args.calib_path, "{:06d}.txt".format(i)))
        img = load_image(osp.join(args.image_path, "{:06d}.png".format(i)))
        ptc = get_ptc_in_image(ptc, calib, img)
        depth_map = get_depth_map(ptc, calib, img, args.collision_policy)
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
            'source_provenance_path': os.path.abspath(args.provenance_json),
            'source_manifest_sha256': digest.hexdigest(),
        }
        with open(osp.join(args.output_path, 'anchor_provenance.json'), 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
