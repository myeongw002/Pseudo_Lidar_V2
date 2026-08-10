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

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
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
SUMMARY_FIELDS = ("selection_mode", "mae", "rmse", "median_abs", "p90_abs", "p95_abs")


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


def read_common_hidden_summary(path, mode):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if row.get("method") == mode and row.get("area") == "common_hidden_valid"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one common_hidden_valid summary for {mode} in {path}, got {len(matches)}"
        )
    return {field: matches[0][field] for field in SUMMARY_FIELDS}


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

    outputs = {}
    for mode in MODES:
        mode_root = compare_root / mode
        range_output = mode_root / "G64_range"
        mask_output = mode_root / "G64_mask"
        stats_csv = mode_root / "meta" / "range_gdc_stats.csv"
        outputs[mode] = range_output
        command = [
            sys.executable, str(REPO_ROOT / "range_gdc" / "range_main_batch.py"),
            "--pred_path", str(raw_range),
            "--anchor_path", str(anchor_range),
            "--output_path", str(range_output),
            "--mask_output_path", str(mask_output),
            "--projection_meta_path", str(projection_meta),
            "--meta_dir", str(mode_root / "meta"),
            "--stats_csv", str(stats_csv),
            "--threads", str(threads),
            "--anchor_reject", str(anchor_filter.get("mode", "abs")),
            "--abs_error_thr", str(anchor_filter.get("abs_error_thr", 2.0)),
            "--log_ratio_thr", str(anchor_filter.get("log_ratio_thr", 0.4)),
            "--anchor_force_policy", str(anchor_filter.get("force_policy", range_cfg.get("anchor_force_policy", "accepted_only"))),
            "--ablation_mode", "full",
            "--selection_mode", mode,
        ]
        for name in RANGE_GDC_OPTIONS:
            if name in range_cfg:
                add_config_arg(command, name, range_cfg[name])
        # selection_mode is intentionally the sole changed Range-GDC parameter.
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
        "--range_min", str(range_cfg.get("range_min", 0.1)),
        "--range_max", str(range_cfg.get("range_max", 80.0)),
    ]
    print("Evaluating:", " ".join(evaluator_command))
    subprocess.run(evaluator_command, check=True)

    summary_rows = [read_common_hidden_summary(evaluator_summary, mode) for mode in MODES]
    summary_path = compare_root / "fusion_comparison_summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved comparable common_hidden_valid metrics: {summary_path}")


if __name__ == "__main__":
    main()
