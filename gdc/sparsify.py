import argparse
import hashlib
import json
import os
import os.path as osp

import numpy as np
from tqdm.auto import tqdm

from data_utils.kitti_object import *
from data_utils.kitti_util import rotz, Calibration, load_image, load_velo_scan
from multiprocessing import Pool

def pto_ang_map(velo_points, H=64, W=512, slice=1, line_spec=None,
                get_lines=False, fill_in_line=None, fill_in_spec=None,
                fill_in_slice=None):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:,1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    if fill_in_line is not None:
        if fill_in_spec is not None:
            depth_map[fill_in_spec] = fill_in_line
        else:
            depth_map[::fill_in_slice, :, :] = fill_in_line

    if line_spec is not None:
        depth_map = depth_map[line_spec, :, :]
    else:
        depth_map = depth_map[::slice, :, :]

    if get_lines:
        depth_map_lines = depth_map.copy()
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]

    if get_lines:
        return depth_map_lines, depth_map
    else:
        return depth_map


def gen_sparse_points(data_idx, args):
    calib = Calibration(osp.join(args.calib_path, "{:06d}.txt".format(data_idx)))
    source_path = osp.join(args.ptc_path, "{:06d}.bin".format(data_idx))
    pc_velo = load_velo_scan(source_path)
    input_count = int(pc_velo.shape[0])
    img = load_image(osp.join(args.image_path, "{:06d}.png".format(data_idx)))
    img_height, img_width, img_channel = img.shape

    if args.image_fov_only:
        _, _, valid_inds_fov = get_lidar_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[valid_inds_fov]

    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]

    if args.fill_in_map_dir is not None and (args.fill_in_spec is not None or args.fill_in_slice is not None):
        fill_in_line = np.load(os.path.join(args.fill_in_map_dir, "{:06d}.npy".format(data_idx)))
    else:
        fill_in_line = None

    if args.store_line_map_dir is not None:
        depth_map_lines, ptc = pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice,\
                    line_spec=args.line_spec, get_lines=True,\
                    fill_in_line=fill_in_line, fill_in_spec=args.fill_in_spec,
                    fill_in_slice=args.fill_in_slice)
        np.save(osp.join(args.store_line_map_dir,
                            "{:06d}".format(data_idx)), depth_map_lines)
        return ptc, input_count, source_path
    else:
        ptc = pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice,\
                            line_spec=args.line_spec, get_lines=False,
                            fill_in_line=fill_in_line, fill_in_spec=args.fill_in_spec,
                            fill_in_slice=args.fill_in_slice)
        return ptc, input_count, source_path


def sparse_and_save(args, data_idx):
    sparse_points, input_count, source_path = gen_sparse_points(data_idx, args)
    sparse_points = sparse_points.astype(np.float32)
    output_path = args.output_path + '/' + '%06d.bin' % (data_idx)
    sparse_points.tofile(output_path)
    digest = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "frame_id": f"{data_idx:06d}",
        "source_velodyne_path": osp.abspath(source_path),
        "input_point_count": input_count,
        "sparse_point_count": int(sparse_points.shape[0]),
        "output_path": osp.abspath(output_path),
        "sha256": digest.hexdigest(),
    }

def gen_sparse_points_all(args):
    with open(args.split_file) as f:
        data_idx_list = [int(x.strip())
                    for x in f.readlines() if len(x.strip()) > 0]

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.store_line_map_dir is not None and not osp.exists(args.store_line_map_dir):
        os.makedirs(args.store_line_map_dir)

    tasks = [(args, data_idx) for data_idx in data_idx_list]
    if args.threads <= 1:
        frames = [sparse_and_save(*task) for task in tqdm(tasks)]
    else:
        with Pool(args.threads) as pool:
            frames = list(tqdm(pool.starmap(sparse_and_save, tasks), total=len(tasks)))
    frames.sort(key=lambda item: item["frame_id"])
    manifest_hash = hashlib.sha256(
        "".join(item["sha256"] for item in frames).encode("ascii")
    ).hexdigest()
    provenance_path = args.provenance_json or osp.join(
        osp.dirname(osp.normpath(args.output_path)), "provenance.json"
    )
    payload = {
        "schema_version": "shared_sparse_anchor_v1",
        "source_velodyne_dir": osp.abspath(args.ptc_path),
        "selected_line_indices": args.line_spec,
        "extraction_height": int(args.H),
        "extraction_width": int(args.W),
        "angular_extraction": {
            "vertical_step_deg": float(0.4 * 64.0 / args.H),
            "horizontal_span_deg": 90.0,
            "horizontal_step_deg": float(90.0 / args.W),
            "slice": int(args.slice),
        },
        "image_fov_only": bool(args.image_fov_only),
        "output_directory": osp.abspath(args.output_path),
        "frame_count": len(frames),
        "manifest_sha256": manifest_hash,
        "frames": frames,
    }
    parent = osp.dirname(provenance_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(provenance_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate sparse pseudo-LiDAR points")
    parser.add_argument('--calib_path', type=str,
                        help='path to calibration files')
    parser.add_argument('--image_path', type=str,
                        help='path to image files')
    parser.add_argument('--ptc_path', type=str,
                        help='path to point cloud files')
    parser.add_argument('--output_path', type=str,
                        help='path to sparsed point cloud files')
    parser.add_argument('--slice', default=1, type=int)
    parser.add_argument('--H', default=64, type=int)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=700, type=int)
    parser.add_argument('--store_line_map_dir', type=str, default=None)
    parser.add_argument('--line_spec', type=int, nargs='+', default=None)
    parser.add_argument('--fill_in_map_dir', type=str, default=None)
    parser.add_argument('--fill_in_spec', type=int,
                        nargs='+', default=None)
    parser.add_argument('--fill_in_slice', type=int, default=None)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--image_fov_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--provenance_json', default=None)
    args = parser.parse_args()

    gen_sparse_points_all(args)
