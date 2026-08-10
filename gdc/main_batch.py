'''
Perform Graph-based Depth Correction (GDC)
in batch over KITTI object dataset.

Author: Yurong You
Date: Feb 2020
'''

import argparse
import csv
import os
import os.path as osp
import time
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

from data_utils.kitti_util import Calibration
from gdc import GDC, anchor_accept_mask, depth2ptc, filter_mask, filter_theta_mask

parser = argparse.ArgumentParser(description='GDC in batch')
parser.add_argument('--input_path', type=str,
                    help='path to predicted depthmap')
parser.add_argument('--calib_path', type=str,
                    help='path to calibration files')
parser.add_argument('--gt_depthmap_path', type=str,
                    help='path to groundtruth depthmap')
parser.add_argument('--output_path', type=str)
parser.add_argument('--split_file', type=str, required=True,
                    help='indices of scene to be corrected')
parser.add_argument('--k', type=int, default=10, help="k for KNN")
parser.add_argument('--recon_tol', type=float, default=5e-4, help="recon_tol for GDC")
parser.add_argument('--method', type=str, default='cg',
                    help='cg or gmres')
parser.add_argument('--disable_subsample', dest="subsample", action='store_false',
                    help='whether subsampling points')
parser.add_argument('--consider_range', type=float, nargs='+', default=[-0.1, 3.0],
                    help='consider_range')
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--stats_csv', type=str, default=None,
                    help='optional image GDC diagnostic CSV path')
parser.add_argument('--anchor_reject', choices=['none', 'abs', 'log_ratio'], default='abs')
parser.add_argument('--abs_error_thr', type=float, default=2.0)
parser.add_argument('--log_ratio_thr', type=float, default=0.4)
parser.add_argument(
    '--subsample_strategy',
    choices=['legacy_random', 'seeded_random', 'deterministic'],
    default='legacy_random',
)
parser.add_argument('--subsample_seed', type=int, default=0)
parser.add_argument(
    '--subsample_output',
    choices=['preserve', 'sparse'],
    default='preserve',
    help='preserve keeps unprocessed prediction pixels; sparse reproduces the legacy sparse-only output',
)
parser.add_argument(
    '--anchor_force_policy',
    choices=['accepted_only', 'all_valid', 'none'],
    default='accepted_only',
    help='which sparse anchors are copied into the final corrected depth map',
)
parser.add_argument('--overwrite', action='store_true')


STATS_FIELDNAMES = [
    "scene_id",
    "status",
    "N_sparse_depth_total",
    "N_sparse_depth_valid",
    "anchor_candidate_count",
    "anchor_overlap_count",
    "anchor_before_reject_count",
    "anchor_after_reject_count",
    "anchor_reject_count",
    "anchor_reject_ratio",
    "anchor_reject_mode",
    "anchor_force_policy",
    "anchor_forced_count",
    "abs_error_thr",
    "log_ratio_thr",
    "raw_anchor_mae_before_gdc",
    "raw_anchor_rmse_before_gdc",
    "corrected_anchor_mae_after_gdc",
    "corrected_anchor_rmse_after_gdc",
    "corrected_anchor_mae",
    "corrected_anchor_rmse",
    "accepted_anchor_mae_after_gdc",
    "accepted_anchor_rmse_after_gdc",
    "rejected_anchor_mae_after_gdc",
    "rejected_anchor_rmse_after_gdc",
    "output_valid_pixels",
    "output_valid_ratio",
    "depth_min",
    "depth_median",
    "depth_max",
    "correction_time_sec",
    "correction_node_count",
    "subsampling_enabled",
    "subsample_strategy",
    "subsample_seed",
    "subsample_output",
]


def GDC_and_save(func, save_path, *args, **kwds):
    started = time.perf_counter()
    result = func(*args, **kwds)
    correction_time = time.perf_counter() - started
    if isinstance(result, tuple):
        corrected, core_stats = result
    else:
        corrected, core_stats = result, {}
    core_stats["correction_time_sec"] = correction_time
    np.save(save_path, corrected.astype(np.float32))
    return corrected, core_stats


def _mean_abs_rmse(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values * values)))


def image_gdc_stats(idx, predict, gt, corrected, calib, consider_range,
                    anchor_reject, abs_error_thr, log_ratio_thr,
                    anchor_force_policy, core_stats):
    ptc_pred = depth2ptc(predict, calib)
    consider_pl = (
        filter_mask(ptc_pred)
        * filter_theta_mask(
            ptc_pred,
            low=np.radians(consider_range[0]),
            high=np.radians(consider_range[1]),
        )
    ).reshape(predict.shape)

    ptc_gt = depth2ptc(gt, calib)
    sparse_total = gt > 0
    sparse_valid = (filter_mask(ptc_gt).reshape(gt.shape)) & sparse_total
    anchor_before = sparse_valid & consider_pl
    accepted = anchor_accept_mask(
        predict, gt, anchor_before, anchor_reject, abs_error_thr, log_ratio_thr
    )

    before_err = predict[anchor_before] - gt[anchor_before]
    after_err = corrected[anchor_before] - gt[anchor_before]
    accepted_after_err = corrected[accepted] - gt[accepted]
    rejected = anchor_before & ~accepted
    rejected_after_err = corrected[rejected] - gt[rejected]
    before_mae, before_rmse = _mean_abs_rmse(before_err)
    after_mae, after_rmse = _mean_abs_rmse(after_err)
    accepted_after_mae, accepted_after_rmse = _mean_abs_rmse(accepted_after_err)
    rejected_after_mae, rejected_after_rmse = _mean_abs_rmse(rejected_after_err)

    valid_output = np.isfinite(corrected) & (corrected > 0)
    valid_values = corrected[valid_output]
    if valid_values.size:
        depth_min = float(np.min(valid_values))
        depth_median = float(np.median(valid_values))
        depth_max = float(np.max(valid_values))
    else:
        depth_min = depth_median = depth_max = np.nan

    before_count = int(anchor_before.sum())
    after_count = int(accepted.sum())
    return {
        "scene_id": f"{idx:06d}",
        "status": "ok",
        "N_sparse_depth_total": int(sparse_total.sum()),
        "N_sparse_depth_valid": int(sparse_valid.sum()),
        "anchor_candidate_count": int(sparse_valid.sum()),
        "anchor_overlap_count": before_count,
        "anchor_before_reject_count": before_count,
        "anchor_after_reject_count": after_count,
        "anchor_reject_count": before_count - after_count,
        "anchor_reject_ratio": (
            (before_count - after_count) / before_count if before_count else np.nan
        ),
        "anchor_reject_mode": anchor_reject,
        "anchor_force_policy": anchor_force_policy,
        "abs_error_thr": float(abs_error_thr),
        "log_ratio_thr": float(log_ratio_thr),
        "raw_anchor_mae_before_gdc": before_mae,
        "raw_anchor_rmse_before_gdc": before_rmse,
        "corrected_anchor_mae_after_gdc": after_mae,
        "corrected_anchor_rmse_after_gdc": after_rmse,
        "corrected_anchor_mae": after_mae,
        "corrected_anchor_rmse": after_rmse,
        "accepted_anchor_mae_after_gdc": accepted_after_mae,
        "accepted_anchor_rmse_after_gdc": accepted_after_rmse,
        "rejected_anchor_mae_after_gdc": rejected_after_mae,
        "rejected_anchor_rmse_after_gdc": rejected_after_rmse,
        "output_valid_pixels": int(valid_output.sum()),
        "output_valid_ratio": float(valid_output.sum() / valid_output.size),
        "depth_min": depth_min,
        "depth_median": depth_median,
        "depth_max": depth_max,
        **core_stats,
    }


def _run_gdc_task(task):
    (idx, input_path, gt_depthmap_path, calib_path, output_path,
     recon_tol, k, method, subsample, consider_range, anchor_reject,
     abs_error_thr, log_ratio_thr, subsample_strategy, subsample_seed,
     subsample_output, anchor_force_policy, overwrite) = task

    save_path = osp.join(output_path, "{:06d}".format(idx))
    if (not overwrite) and osp.exists(save_path + '.npy'):
        return idx, {"scene_id": f"{idx:06d}", "status": "skipped"}

    predict = np.load(osp.join(input_path, "{:06d}.npy".format(idx)))
    gt = np.load(osp.join(gt_depthmap_path, "{:06d}.npy".format(idx)))
    calib = Calibration(osp.join(calib_path, "{:06d}.txt".format(idx)))

    frame_seed = int(subsample_seed) + int(idx)
    corrected, core_stats = GDC_and_save(
        GDC, save_path, predict, gt, calib,
        W_tol=1e-5,
        recon_tol=recon_tol,
        k=k,
        method=method,
        subsample=subsample,
        consider_range=consider_range,
        anchor_reject=anchor_reject,
        abs_error_thr=abs_error_thr,
        log_ratio_thr=log_ratio_thr,
        subsample_strategy=subsample_strategy,
        subsample_seed=frame_seed,
        subsample_output=subsample_output,
        anchor_force_policy=anchor_force_policy,
        return_stats=True,
    )
    core_stats.update({
        "subsampling_enabled": bool(subsample),
        "subsample_strategy": subsample_strategy,
        "subsample_seed": frame_seed,
        "subsample_output": subsample_output,
        "anchor_force_policy": anchor_force_policy,
    })
    return idx, image_gdc_stats(
        idx, predict, gt, corrected, calib, consider_range,
        anchor_reject, abs_error_thr, log_ratio_thr,
        anchor_force_policy, core_stats
    )


def write_stats_csv(path, rows):
    if not path:
        return
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(args):
    if not osp.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(args.split_file) as f:
        idx_list = [int(x.strip()) for x in f.readlines() if len(x.strip()) > 0]

    tasks = [
        (idx, args.input_path, args.gt_depthmap_path, args.calib_path,
         args.output_path, args.recon_tol, args.k, args.method,
         args.subsample, args.consider_range, args.anchor_reject,
         args.abs_error_thr, args.log_ratio_thr, args.subsample_strategy,
         args.subsample_seed, args.subsample_output,
         args.anchor_force_policy, args.overwrite)
        for idx in idx_list
    ]

    stats_rows = []
    if args.threads <= 1:
        for task in tqdm(tasks):
            _, stats = _run_gdc_task(task)
            stats_rows.append(stats)
    else:
        with Pool(args.threads) as pool:
            for _, stats in tqdm(
                pool.imap_unordered(_run_gdc_task, tasks, chunksize=1),
                total=len(tasks)
            ):
                stats_rows.append(stats)
    write_stats_csv(args.stats_csv, sorted(stats_rows, key=lambda row: row["scene_id"]))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
