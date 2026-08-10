#!/usr/bin/env python3
"""Compare confidence-hard and always-soft Range-GDC fusion on saved inputs.

This runner deliberately does not invoke the SDN, anchor-generation, or GT
projection stages.  It reads those artifacts from a completed canonical
pipeline output root and creates an isolated comparison directory.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.evaluate_range_metrics import area_supports, load_projection_meta, valid_range_mask
from range_gdc.range_projection import find_input_npy


MODES = ("confidence_hard", "soft")
RANGE_GDC_OPTIONS = (
    "method", "range_min", "range_max", "lambda_anchor", "lambda_prior",
    "lambda_smooth", "neighbor", "edge_spatial_mode", "sigma_angular",
    "sigma_tangent", "sigma_log_range", "max_log_range_diff", "transfer_k",
    "transfer_neighbor_mode", "direct_weight_mode", "confidence_mode",
    "sigma_conf_pixel", "sigma_conf_angular", "sigma_conf_log_range",
    "confidence_power", "confidence_min", "confidence_max",
    "confidence_high_thr", "confidence_low_thr", "direct_log_range_thr",
    "graph_log_range_thr", "delta_clip",
)
SUMMARY_FIELDS = ("selection_mode", "mae", "rmse", "median_abs", "p90_abs", "p95_abs", "eval_pixels")


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path):
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def add_config_arg(command, name, value):
    if value is None:
        return
    command.extend([f"--{name}", str(value)])


def require_directory(path, label):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def build_range_gdc_parameters(range_cfg, anchor_filter):
    """Resolve every comparison parameter once so both modes share it exactly."""
    defaults = {
        "anchor_reject": "abs", "abs_error_thr": 2.0, "log_ratio_thr": 0.4,
        "anchor_force_policy": "accepted_only", "method": "cg", "range_min": 0.1,
        "range_max": 80.0, "lambda_anchor": 300.0, "lambda_prior": 0.1,
        "lambda_smooth": 1.0, "neighbor": "angular_grid8", "edge_spatial_mode": "angular",
        "sigma_angular": 0.01, "sigma_tangent": 1.0, "sigma_log_range": 0.3,
        "max_log_range_diff": None, "transfer_k": 1, "transfer_neighbor_mode": "rowcol",
        "direct_weight_mode": "nearest", "confidence_mode": "nearest",
        "sigma_conf_pixel": 2.0, "sigma_conf_angular": 0.01,
        "sigma_conf_log_range": 0.05, "confidence_power": 2.0,
        "confidence_min": 0.0, "confidence_max": 1.0, "confidence_high_thr": 0.8,
        "confidence_low_thr": 0.2, "direct_log_range_thr": 0.05,
        "graph_log_range_thr": 0.2, "delta_clip": 0.3,
    }
    parameters = dict(defaults)
    parameters.update({name: range_cfg[name] for name in RANGE_GDC_OPTIONS if name in range_cfg})
    parameters.update({
        "anchor_reject": anchor_filter.get("mode", parameters["anchor_reject"]),
        "abs_error_thr": anchor_filter.get("abs_error_thr", parameters["abs_error_thr"]),
        "log_ratio_thr": anchor_filter.get("log_ratio_thr", parameters["log_ratio_thr"]),
        "anchor_force_policy": anchor_filter.get(
            "force_policy", range_cfg.get("anchor_force_policy", parameters["anchor_force_policy"])
        ),
        "ablation_mode": "full",
    })
    return parameters


def assert_parameter_identity(parameters_by_mode):
    """Fail loudly if a future mode-specific edit changes anything but selection."""
    reference_mode = MODES[0]
    reference = parameters_by_mode[reference_mode]
    for mode in MODES[1:]:
        candidate = parameters_by_mode[mode]
        keys = set(reference) | set(candidate)
        differences = {
            key: (reference.get(key), candidate.get(key))
            for key in keys
            if key != "selection_mode" and reference.get(key) != candidate.get(key)
        }
        if differences:
            raise RuntimeError(
                f"Fusion comparison parameters differ between {reference_mode} and {mode}: {differences}"
            )


def build_range_gdc_command(parameters, *, raw_range, anchor_range, split_file, projection_meta, mode_root, threads):
    command = [
        sys.executable, str(REPO_ROOT / "range_gdc" / "range_main_batch.py"),
        "--pred_path", str(raw_range), "--anchor_path", str(anchor_range),
        "--split_file", str(split_file), "--output_path", str(mode_root / "G64_range"),
        "--mask_output_path", str(mode_root / "G64_mask"),
        "--projection_meta_path", str(projection_meta), "--meta_dir", str(mode_root / "meta"),
        "--stats_csv", str(mode_root / "meta" / "range_gdc_stats.csv"),
        "--threads", str(threads),
    ]
    for name in ("anchor_reject", "abs_error_thr", "log_ratio_thr", "anchor_force_policy", "ablation_mode") + RANGE_GDC_OPTIONS + ("selection_mode",):
        add_config_arg(command, name, parameters[name])
    return command


def aggregate_common_hidden_metrics(split_file, gt_range, guide_range, source_range, method_paths, *, range_min, range_max, selected_rows, projection_meta_path):
    """Aggregate pixel errors using the evaluator's exact common-hidden mask."""
    projection_meta = load_projection_meta(projection_meta_path)
    evaluation_args = SimpleNamespace(
        range_min=float(range_min), range_max=float(range_max), source_rows=32,
        row_offset=None, row_stride=None, source_row_indices=np.asarray(selected_rows, dtype=np.int32),
        selected_rows_dir=None,
        projection_selected_rows=(
            projection_meta["selected_rows"].astype(np.int32)
            if projection_meta is not None and "selected_rows" in projection_meta.files
            else None
        ),
    )
    errors_by_mode = {mode: [] for mode in method_paths}
    eval_pixels = 0
    with open(split_file) as handle:
        scene_ids = [int(line.strip()) for line in handle if line.strip()]
    for scene_id in scene_ids:
        gt = np.load(find_input_npy(str(gt_range), scene_id)).astype(np.float32)
        guide = np.load(find_input_npy(str(guide_range), scene_id)).astype(np.float32)
        source = np.load(find_input_npy(str(source_range), scene_id)).astype(np.float32)
        supports, _, _, gt_valid = area_supports(scene_id, gt, guide, source, evaluation_args)
        predictions = {
            mode: np.load(find_input_npy(str(path), scene_id)).astype(np.float32)
            for mode, path in method_paths.items()
        }
        if any(prediction.shape != gt.shape for prediction in predictions.values()):
            raise ValueError(f"{scene_id:06d}: prediction shape differs from GT shape")
        all_pred_valid = np.ones_like(gt_valid, dtype=bool)
        for prediction in predictions.values():
            all_pred_valid &= valid_range_mask(prediction, range_min, range_max)
        common_hidden_valid = supports["hidden_rows_valid"] & gt_valid & all_pred_valid
        eval_pixels += int(common_hidden_valid.sum())
        for mode, prediction in predictions.items():
            errors_by_mode[mode].append(
                prediction[common_hidden_valid].astype(np.float64) - gt[common_hidden_valid].astype(np.float64)
            )

    summary_rows = []
    for mode in MODES:
        errors = np.concatenate(errors_by_mode[mode]) if errors_by_mode[mode] else np.array([], dtype=np.float64)
        if not errors.size:
            raise RuntimeError("common_hidden_valid contains no pixels; cannot compare fusion modes")
        abs_errors = np.abs(errors)
        summary_rows.append({
            "selection_mode": mode,
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "median_abs": float(np.median(abs_errors)),
            "p90_abs": float(np.percentile(abs_errors, 90)),
            "p95_abs": float(np.percentile(abs_errors, 95)),
            "eval_pixels": int(eval_pixels),
        })
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/r64_pipeline_test_1000.yaml")
    parser.add_argument(
        "--pipeline-output-root",
        default=None,
        help="Override config output_root containing the already-generated inputs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Comparison root; default is <pipeline-output-root>/range/fusion_compare.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly reuse existing comparison predictions; no files are overwritten.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve(args.config)
    cfg = load_yaml(config_path)
    pipeline_root = Path(args.pipeline_output_root or cfg["output_root"]).expanduser()
    compare_root = Path(args.output_root).expanduser() if args.output_root else pipeline_root / "range" / "fusion_compare"
    split_file = resolve(cfg["split_file"])
    range_cfg = dict(cfg.get("range_gdc", {}))
    anchor_filter = dict(cfg.get("anchor_filter", {}))
    range_anchor_cfg = dict(cfg.get("range_anchor", {}))
    selected_rows = [int(value) for value in range_anchor_cfg.get("selected_rows", [5, 7, 9, 11])]
    threads = int(args.threads if args.threads is not None else range_cfg.get("threads", cfg.get("threads", 1)))
    if threads <= 0:
        raise ValueError("--threads must be positive")

    raw_range = pipeline_root / "range" / "raw_sdn" / "G64_range"
    anchor_range = pipeline_root / "anchor" / "range" / "G64_range"
    gt_range = pipeline_root / "range" / "gt" / "G64_range"
    projection_meta = pipeline_root / "range" / "gt" / "meta" / "projection_meta.npz"
    for path, label in ((raw_range, "raw SDN range"), (anchor_range, "Range-GDC anchor"), (gt_range, "GT range")):
        require_directory(path, label)
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    if not projection_meta.is_file():
        raise FileNotFoundError(f"Missing projection metadata: {projection_meta}")

    if compare_root.exists() and any(compare_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to reuse existing comparison outputs: {compare_root}. "
            "Choose a new --output-root or pass --resume explicitly."
        )
    compare_root.mkdir(parents=True, exist_ok=True)

    base_parameters = build_range_gdc_parameters(range_cfg, anchor_filter)
    parameters_by_mode = {
        mode: {**base_parameters, "selection_mode": mode}
        for mode in MODES
    }
    assert_parameter_identity(parameters_by_mode)

    outputs = {}
    for mode in MODES:
        mode_root = compare_root / mode
        range_output = mode_root / "G64_range"
        outputs[mode] = range_output
        command = build_range_gdc_command(
            parameters_by_mode[mode], raw_range=raw_range, anchor_range=anchor_range,
            split_file=split_file, projection_meta=projection_meta, mode_root=mode_root,
            threads=threads,
        )
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

    metrics_csv = compare_root / "fusion_comparison_metrics.csv"
    evaluator_summary = compare_root / "fusion_comparison_evaluator_summary.csv"
    evaluator_command = [
        sys.executable, str(REPO_ROOT / "range_gdc" / "evaluate_range_metrics.py"),
        "--split_file", str(split_file),
        "--gt_range_path", str(gt_range),
        "--guide_range_path", str(anchor_range),
        "--source_range_path", str(raw_range),
        "--method", f"confidence_hard={outputs['confidence_hard']}",
        "--method", f"soft={outputs['soft']}",
        "--metrics_csv", str(metrics_csv),
        "--summary_csv", str(evaluator_summary),
        "--projection_meta_path", str(projection_meta),
        "--anchor_rows", *[str(row) for row in selected_rows],
        "--range_min", str(base_parameters["range_min"]),
        "--range_max", str(base_parameters["range_max"]),
    ]
    print("Evaluating:", " ".join(evaluator_command))
    subprocess.run(evaluator_command, check=True)

    summary_rows = aggregate_common_hidden_metrics(
        split_file, gt_range, anchor_range, raw_range, outputs,
        range_min=base_parameters["range_min"], range_max=base_parameters["range_max"],
        selected_rows=selected_rows, projection_meta_path=projection_meta,
    )
    summary_path = compare_root / "fusion_comparison_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved comparable common_hidden_valid metrics: {summary_path}")


if __name__ == "__main__":
    main()
