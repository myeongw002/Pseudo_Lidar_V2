"""Batch runner for graph-only Range-GDC."""

import argparse
import csv
import os
import os.path as osp
import re
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

try:
    from .range_gdc import RangeROIGDC
except ImportError:
    from range_gdc import RangeROIGDC


STATS_FIELDNAMES = [
    "scene_id",
    "status",
    "H",
    "W",
    "N_valid_pred",
    "N_nodes",
    "N_anchor_valid",
    "N_anchor_overlap",
    "N_residual_targets",
    "N_rejected_residual_targets",
    "anchor_candidate_count",
    "anchor_overlap_count",
    "anchor_before_reject_count",
    "anchor_after_reject_count",
    "anchor_reject_count",
    "anchor_reject_ratio",
    "anchor_reject_mode",
    "anchor_force_policy",
    "anchor_forced_count",
    "output_valid_count",
    "output_valid_ratio",
    "method_tag",
    "residual_domain",
    "delta_graph_mean",
    "delta_graph_std",
    "delta_graph_abs_mean",
    "propagation_ratio_graph",
    "delta_final_mean",
    "delta_final_std",
    "delta_final_abs_mean",
    "neighbor",
    "edge_spatial_mode",
    "edge_range_mode",
    "sigma_angular",
    "sigma_tangent",
    "sigma_log_range",
    "max_log_range_diff",
    "N_edges_graph",
    "edge_weight_mean",
    "edge_weight_std",
    "edge_weight_min",
    "edge_weight_max",
    "edge_weight_zero_count",
    "edge_weight_zero_ratio",
    "range_gate_mean",
    "angular_weight_mean",
    "spatial_distance_mean",
    "log_range_diff_mean",
    "lambda_anchor",
    "lambda_prior",
    "lambda_smooth",
    "delta_clip",
    "residual_target_mean",
    "residual_target_std",
    "residual_target_abs_mean",
    "anchor_abs_error_before",
    "anchor_rmse_before",
    "anchor_abs_error_after_graph_solve",
    "anchor_rmse_after_graph_solve",
    "anchor_abs_error_after_graph",
    "anchor_rmse_after_graph",
    "anchor_abs_error_after_force",
    "anchor_rmse_after_force",
    "accepted_anchor_mae_after_correction",
    "accepted_anchor_rmse_after_correction",
    "rejected_anchor_mae_after_correction",
    "rejected_anchor_rmse_after_correction",
    "all_overlap_anchor_mae_after_correction",
    "all_overlap_anchor_rmse_after_correction",
    "solver_info",
    "solve_residual",
    "t_build_nodes",
    "t_build_graph",
    "t_graph_solve",
    "t_total_correction",
]


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


def npy_map(directory):
    files = sorted(glob(osp.join(directory, "*.npy")))
    mapping = {}
    for path in files:
        mapping.setdefault(normalize_scene_id(path), path)
    return mapping


def discover_scene_ids(pred_path, anchor_path):
    pred = npy_map(pred_path)
    anchor = npy_map(anchor_path)
    common = sorted(set(pred) & set(anchor))
    if not common:
        raise RuntimeError(f"No common .npy scene ids found between {pred_path} and {anchor_path}")
    return common, pred, anchor


def read_split_scene_ids(split_file):
    """Read a split as unique, zero-padded scene ids in deterministic order."""
    with open(split_file) as handle:
        ids = [f"{int(line.strip()):06d}" for line in handle if line.strip()]
    if not ids:
        raise ValueError(f"No scene ids in split file: {split_file}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate scene ids in split file: {split_file}")
    return sorted(ids)


def select_scene_ids(pred_files, anchor_files, split_file=None):
    """Select common files, optionally requiring every id from a split."""
    if split_file is None:
        scene_ids = sorted(set(pred_files) & set(anchor_files))
        if not scene_ids:
            raise RuntimeError("No common .npy scene ids found between prediction and anchor directories")
        return scene_ids

    scene_ids = read_split_scene_ids(split_file)
    missing_pred = [scene_id for scene_id in scene_ids if scene_id not in pred_files]
    missing_anchor = [scene_id for scene_id in scene_ids if scene_id not in anchor_files]
    if missing_pred or missing_anchor:
        parts = []
        if missing_pred:
            parts.append(f"missing prediction frames ({len(missing_pred)}): {missing_pred[:5]}")
        if missing_anchor:
            parts.append(f"missing anchor frames ({len(missing_anchor)}): {missing_anchor[:5]}")
        raise FileNotFoundError(f"Split {split_file} cannot be processed: " + "; ".join(parts))
    return scene_ids


def scalar_from_npz(meta, key, default=None):
    if key not in meta:
        return default
    value = meta[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def find_projection_meta(meta_dir):
    candidates = sorted(glob(osp.join(meta_dir, "*_projection_meta.npz")))
    if not candidates:
        raise FileNotFoundError(f"No *_projection_meta.npz files found in {meta_dir}")
    return candidates[0]


def load_meta_info(args):
    if args.projection_meta_path is not None:
        meta_path = args.projection_meta_path
    elif args.meta_dir is not None:
        meta_path = find_projection_meta(args.meta_dir)
    else:
        return {}

    meta = np.load(meta_path, allow_pickle=True)
    info = {
        "meta_path": meta_path,
        "height": scalar_from_npz(meta, "height"),
        "width": scalar_from_npz(meta, "width"),
        "azimuth_mode": scalar_from_npz(meta, "azimuth_mode", "full_360_front_centered"),
    }
    if "vertical_centers_deg" in meta.files:
        info["vertical_centers_deg"] = meta["vertical_centers_deg"].astype(np.float64)
    if "azimuth_centers_deg" in meta.files:
        info["azimuth_centers_deg"] = meta["azimuth_centers_deg"].astype(np.float64)
    return info


def corrected_output_paths(output_path, mask_output_path, scene_id, H):
    guide_name = f"G{H}"
    return (
        osp.join(output_path, f"{scene_id}_{guide_name}_corr_range.npy"),
        osp.join(mask_output_path, f"{scene_id}_{guide_name}_corr_mask.npy"),
    )


def process_one(task):
    (
        scene_id,
        pred_file,
        anchor_file,
        output_path,
        mask_output_path,
        projection_shape,
        vertical_centers_deg,
        azimuth_centers_deg,
        azimuth_mode,
        args_dict,
    ) = task

    pred = np.load(pred_file).astype(np.float32)
    anchor_input = np.load(anchor_file).astype(np.float32)
    if pred.ndim != 2:
        raise ValueError(f"{scene_id}: pred range must be 2D, got {pred.shape}")
    if anchor_input.ndim != 2:
        raise ValueError(f"{scene_id}: canonical Range-GDC anchor must be 2D, got {anchor_input.shape}")
    if projection_shape is not None and pred.shape != projection_shape:
        raise ValueError(
            f"{scene_id}: prediction shape {pred.shape} does not match "
            f"projection metadata shape {projection_shape}"
        )
    if anchor_input.shape != pred.shape:
        raise ValueError(
            f"{scene_id}: canonical Range-GDC anchor shape {anchor_input.shape} "
            f"does not match prediction shape {pred.shape}"
        )
    anchor = anchor_input

    range_file, mask_file = corrected_output_paths(output_path, mask_output_path, scene_id, pred.shape[0])
    if (not args_dict["overwrite"]) and osp.exists(range_file) and osp.exists(mask_file):
        return scene_id, "skipped", {"scene_id": scene_id, "status": "skipped"}

    corrected, mask, stats = RangeROIGDC(
        pred_range=pred,
        anchor_range=anchor,
        vertical_centers_deg=vertical_centers_deg,
        azimuth_centers_deg=azimuth_centers_deg,
        azimuth_mode=azimuth_mode,
        method=args_dict["method"],
        range_min=args_dict["range_min"],
        range_max=args_dict["range_max"],
        anchor_reject=args_dict["anchor_reject"],
        log_ratio_thr=args_dict["log_ratio_thr"],
        abs_error_thr=args_dict["abs_error_thr"],
        lambda_anchor=args_dict["lambda_anchor"],
        lambda_prior=args_dict["lambda_prior"],
        lambda_smooth=args_dict["lambda_smooth"],
        neighbor=args_dict["neighbor"],
        edge_spatial_mode=args_dict["edge_spatial_mode"],
        sigma_angular=args_dict["sigma_angular"],
        sigma_tangent=args_dict["sigma_tangent"],
        sigma_log_range=args_dict["sigma_log_range"],
        max_log_range_diff=args_dict["max_log_range_diff"],
        edge_range_mode=args_dict["edge_range_mode"],
        residual_domain=args_dict["residual_domain"],
        delta_clip=args_dict["delta_clip"],
        anchor_force_policy=args_dict["anchor_force_policy"],
        return_stats=True,
        verbose=args_dict["verbose"],
    )
    stats["scene_id"] = scene_id

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    np.save(range_file, corrected.astype(np.float32))
    np.save(mask_file, mask.astype(bool))
    return scene_id, "ok", stats


def write_stats_csv(path, stats_rows):
    if not path:
        return
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    extra_fields = sorted({k for row in stats_rows for k in row} - set(STATS_FIELDNAMES))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES + extra_fields)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run graph-only Range-GDC.")
    parser.add_argument("--pred_path", "--guide_path", dest="pred_path", required=True)
    parser.add_argument("--anchor_path", required=True)
    parser.add_argument(
        "--split_file",
        default=None,
        help="Optional split restricting correction to these scene ids; every id must exist in both inputs.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mask_output_path", required=True)
    parser.add_argument("--projection_meta_path", default=None)
    parser.add_argument("--meta_dir", default=None)
    parser.add_argument("--stats_csv", default=None)
    parser.add_argument("--method", choices=["cg", "spsolve"], default="cg")
    parser.add_argument("--range_min", type=float, default=0.1)
    parser.add_argument("--range_max", type=float, default=80.0)
    # Keep standalone execution aligned with the checked-in paper configs.
    parser.add_argument("--anchor_reject", choices=["log_ratio", "abs", "none"], default="abs")
    parser.add_argument("--log_ratio_thr", type=float, default=0.4)
    parser.add_argument("--abs_error_thr", type=float, default=2.0)
    parser.add_argument("--lambda_anchor", type=float, default=300.0)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=1.0)
    parser.add_argument("--neighbor", choices=["angular_grid4", "angular_grid8"], default="angular_grid8")
    parser.add_argument("--edge_spatial_mode", choices=["angular", "tangent"], default="angular")
    parser.add_argument(
        "--edge_range_mode",
        choices=["log_gaussian", "uniform"],
        default="log_gaussian",
    )
    parser.add_argument(
        "--residual_domain", choices=["log", "linear"], default="log"
    )
    parser.add_argument("--sigma_angular", type=float, default=0.01)
    parser.add_argument("--sigma_tangent", type=float, default=1.0)
    parser.add_argument("--sigma_log_range", type=float, default=0.3)
    parser.add_argument("--max_log_range_diff", type=float, default=None)
    parser.add_argument("--delta_clip", type=float, default=0.3)
    parser.add_argument(
        "--disable_delta_clip", action="store_const", const=None,
        dest="delta_clip",
    )
    parser.add_argument(
        "--anchor_force_policy",
        choices=["accepted_only", "all_valid", "none"],
        default="accepted_only",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    for path_name in ("pred_path", "anchor_path"):
        path = getattr(args, path_name)
        if not osp.isdir(path):
            raise FileNotFoundError(path)
    if args.split_file is not None and not osp.isfile(args.split_file):
        raise FileNotFoundError(args.split_file)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("--max_items must be positive when set")
    if args.range_max < args.range_min:
        raise ValueError("--range_max must be >= --range_min")
    if args.lambda_anchor < 0 or args.lambda_prior < 0 or args.lambda_smooth < 0:
        raise ValueError("lambda values must be >= 0")
    if args.sigma_angular <= 0 or args.sigma_tangent <= 0 or args.sigma_log_range <= 0:
        raise ValueError("sigma values must be positive")
    if args.max_log_range_diff is not None and args.max_log_range_diff <= 0:
        raise ValueError("--max_log_range_diff must be positive when set")
    if args.delta_clip is not None and args.delta_clip <= 0:
        raise ValueError("--delta_clip must be positive when set")


def main():
    args = parse_args()
    validate_args(args)

    _, pred_files, anchor_files = discover_scene_ids(args.pred_path, args.anchor_path)
    scene_ids = select_scene_ids(pred_files, anchor_files, args.split_file)
    if args.max_items is not None:
        scene_ids = scene_ids[: args.max_items]

    meta_info = load_meta_info(args)
    meta_height = meta_info.get("height")
    meta_width = meta_info.get("width")
    if (meta_height is None) != (meta_width is None):
        raise ValueError("projection metadata must define both height and width")
    projection_shape = (
        None
        if meta_height is None
        else (int(meta_height), int(meta_width))
    )
    vertical_centers_deg = meta_info.get("vertical_centers_deg")
    azimuth_centers_deg = meta_info.get("azimuth_centers_deg")
    azimuth_mode = str(meta_info.get("azimuth_mode", "full_360_front_centered"))

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.mask_output_path, exist_ok=True)

    args_dict = vars(args).copy()
    tasks = [
        (
            scene_id,
            pred_files[scene_id],
            anchor_files[scene_id],
            args.output_path,
            args.mask_output_path,
            projection_shape,
            vertical_centers_deg,
            azimuth_centers_deg,
            azimuth_mode,
            args_dict,
        )
        for scene_id in scene_ids
    ]

    print("Graph-only Range-GDC batch settings")
    print(f"frames            : {len(tasks)}")
    print(f"pred_path         : {args.pred_path}")
    print(f"anchor_path       : {args.anchor_path}")
    print(f"output_path       : {args.output_path}")
    print(f"mask_output_path  : {args.mask_output_path}")
    print(f"method            : {args.method}")
    print(f"neighbor          : {args.neighbor}")
    print(f"edge_spatial_mode : {args.edge_spatial_mode}")
    print(f"edge_range_mode   : {args.edge_range_mode}")
    print(f"residual_domain   : {args.residual_domain}")
    print(f"delta_clip        : {args.delta_clip}")
    print(f"threads           : {args.threads}")

    stats_rows = []
    if args.threads == 1:
        iterator = (process_one(task) for task in tasks)
        for _, _, stats in tqdm(iterator, total=len(tasks)):
            stats_rows.append(stats)
    else:
        # Workers load arrays themselves; streaming one task at a time avoids
        # queuing large numpy arrays in the parent process.
        with Pool(processes=args.threads) as pool:
            for _, _, stats in tqdm(pool.imap_unordered(process_one, tasks, chunksize=1), total=len(tasks)):
                stats_rows.append(stats)

    write_stats_csv(args.stats_csv, stats_rows)
    if args.stats_csv:
        print(f"Done. Saved RangeGDC stats to: {args.stats_csv}")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
