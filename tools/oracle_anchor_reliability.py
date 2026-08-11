#!/usr/bin/env python3
"""ORACLE / GT-ONLY / DIAGNOSTIC anchor-weight headroom experiment.

This standalone tool consumes existing train1000 range artifacts.  It never
generates SDN, anchor, GT, or Original-GDC inputs, and it is not an inference
method.  GT is used only to score accepted anchors and to evaluate outputs.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install requirements.txt first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.range_gdc import RangeROIGDC, valid_range_mask  # noqa: E402
from range_gdc.range_main_batch import npy_map, read_split_scene_ids  # noqa: E402


DIAGNOSTIC_LABEL = "ORACLE / GT-ONLY / DIAGNOSTIC"
TRAIN1000_SPLIT_NAME = "train_1000_seed2026.txt"
KEEP_RATIOS = (
    ("uniform", 1.00),
    ("oracle_keep90", 0.90),
    ("oracle_keep75", 0.75),
    ("oracle_keep50", 0.50),
    ("oracle_keep25", 0.25),
)
SCORE_FIELDS = [
    "diagnostic_label", "frame_id", "anchor_row", "anchor_col",
    "anchor_range", "predicted_range", "abs_discrepancy_m",
    "abs_log_discrepancy", "target_residual", "hidden_neighbor_count",
    "oracle_known", "oracle_badness",
]


def _path_is_under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_diagnostic_roots(base_output_root, output_root):
    base = Path(base_output_root).resolve()
    output = Path(output_root).resolve()
    if base == output or _path_is_under(output, base) or _path_is_under(base, output):
        raise ValueError("Oracle output_root and base_output_root must be disjoint")


def validate_train1000_split(split_file):
    split = Path(split_file)
    if split.name != TRAIN1000_SPLIT_NAME:
        raise ValueError(
            f"{DIAGNOSTIC_LABEL}: split must be named {TRAIN1000_SPLIT_NAME}; "
            "val/test oracle runs are forbidden"
        )


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def graph_invariant_signature(debug, guide_range, anchor_range, range_min, range_max):
    guide_valid = valid_range_mask(guide_range, range_min, range_max)
    anchor_valid = valid_range_mask(anchor_range, range_min, range_max)
    rejected_mask = guide_valid & anchor_valid & ~debug["target_mask"]
    return {
        "node_count": int(debug["node_rows"].size),
        "edge_count": int(debug["edge_weight"].size),
        "target_digest": _array_digest(debug["target_mask"]),
        "rejected_digest": _array_digest(rejected_mask),
        "force_digest": _array_digest(debug["force_mask"]),
        "target_delta_digest": _array_digest(debug["target_delta"]),
        "graph_digest": _array_digest(
            debug["edge_i"], debug["edge_j"], debug["edge_weight"]
        ),
    }


def assert_graph_invariants(expected, actual, scene_id, method):
    if expected != actual:
        changed = sorted(key for key in expected if expected[key] != actual.get(key))
        raise RuntimeError(
            f"{scene_id} {method}: oracle changed canonical invariants: {changed}"
        )


def compute_oracle_badness(
    pred_range,
    gt_range,
    source_rows,
    debug,
    *,
    range_min=0.1,
    range_max=80.0,
    delta_clip=0.3,
):
    """Return weighted 1-hop hidden-GT badness aligned with accepted anchors."""
    pred = np.asarray(pred_range, dtype=np.float64)
    gt = np.asarray(gt_range, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/GT shape mismatch: {pred.shape} vs {gt.shape}")

    target_nodes = np.asarray(debug["target_node_indices"], dtype=np.int64)
    target_delta = np.asarray(debug["target_delta"], dtype=np.float64)
    node_rows = np.asarray(debug["node_rows"], dtype=np.int64)
    node_cols = np.asarray(debug["node_cols"], dtype=np.int64)
    node_to_anchor = np.full(node_rows.shape, -1, dtype=np.int64)
    node_to_anchor[target_nodes] = np.arange(target_nodes.size, dtype=np.int64)

    pred_valid = valid_range_mask(pred, range_min, range_max)
    gt_valid = valid_range_mask(gt, range_min, range_max)
    hidden = np.ones(pred.shape, dtype=bool)
    rows = np.asarray(source_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= pred.shape[0]):
        raise ValueError("source_rows must be valid row indices")
    hidden[rows, :] = False
    eligible_hidden_nodes = (
        hidden[node_rows, node_cols]
        & pred_valid[node_rows, node_cols]
        & gt_valid[node_rows, node_cols]
    )

    gt_delta_map = np.log(np.maximum(gt, 1e-6)) - np.log(np.maximum(pred, 1e-6))
    if delta_clip is not None:
        gt_delta_map = np.clip(gt_delta_map, -float(delta_clip), float(delta_clip))
    gt_delta_nodes = gt_delta_map[node_rows, node_cols]

    weighted_error = np.zeros(target_nodes.shape, dtype=np.float64)
    weight_sum = np.zeros(target_nodes.shape, dtype=np.float64)
    neighbor_count = np.zeros(target_nodes.shape, dtype=np.int64)
    edge_i = np.asarray(debug["edge_i"], dtype=np.int64)
    edge_j = np.asarray(debug["edge_j"], dtype=np.int64)
    edge_weight = np.asarray(debug["edge_weight"], dtype=np.float64)

    for anchor_nodes, hidden_nodes in ((edge_i, edge_j), (edge_j, edge_i)):
        anchor_indices = node_to_anchor[anchor_nodes]
        use = (anchor_indices >= 0) & eligible_hidden_nodes[hidden_nodes]
        if not np.any(use):
            continue
        indices = anchor_indices[use]
        weights = edge_weight[use]
        errors = np.abs(target_delta[indices] - gt_delta_nodes[hidden_nodes[use]])
        np.add.at(weighted_error, indices, weights * errors)
        np.add.at(weight_sum, indices, weights)
        np.add.at(neighbor_count, indices, 1)

    known = weight_sum > 0.0
    badness = np.full(target_nodes.shape, np.nan, dtype=np.float64)
    badness[known] = weighted_error[known] / weight_sum[known]
    return badness, neighbor_count, known


def score_frame(scene_id, pred, anchor, gt, source_rows, debug, params):
    badness, neighbor_count, known = compute_oracle_badness(
        pred,
        gt,
        source_rows,
        debug,
        range_min=params["range_min"],
        range_max=params["range_max"],
        delta_clip=params["delta_clip"],
    )
    rows, cols = np.where(debug["target_mask"])
    records = []
    for index, (row, col) in enumerate(zip(rows, cols)):
        anchor_value = float(anchor[row, col])
        pred_value = float(pred[row, col])
        records.append(
            {
                "diagnostic_label": DIAGNOSTIC_LABEL,
                "frame_id": scene_id,
                "anchor_row": int(row),
                "anchor_col": int(col),
                "anchor_range": anchor_value,
                "predicted_range": pred_value,
                "abs_discrepancy_m": abs(anchor_value - pred_value),
                "abs_log_discrepancy": abs(
                    math.log(max(anchor_value, 1e-6)) - math.log(max(pred_value, 1e-6))
                ),
                "target_residual": float(debug["target_delta"][index]),
                "hidden_neighbor_count": int(neighbor_count[index]),
                "oracle_known": bool(known[index]),
                "oracle_badness": float(badness[index]) if known[index] else math.nan,
            }
        )
    return records


def rank_oracle_weights(score_rows, keep_ratios=KEEP_RATIOS):
    """Create deterministic global rank weights; unknown anchors always keep q=1."""
    known = [row for row in score_rows if row["oracle_known"]]
    known.sort(
        key=lambda row: (
            float(row["oracle_badness"]),
            str(row["frame_id"]),
            int(row["anchor_row"]),
            int(row["anchor_col"]),
        )
    )
    all_keys = [
        (str(row["frame_id"]), int(row["anchor_row"]), int(row["anchor_col"]))
        for row in score_rows
    ]
    unknown_keys = {
        key for key, row in zip(all_keys, score_rows) if not row["oracle_known"]
    }
    weights = {}
    threshold_rows = []
    for method, ratio in keep_ratios:
        keep_count = len(known) if method == "uniform" else int(math.ceil(ratio * len(known)))
        kept = known[:keep_count]
        kept_keys = {
            (str(row["frame_id"]), int(row["anchor_row"]), int(row["anchor_col"]))
            for row in kept
        }
        method_weights = {
            key: 1.0 if key in unknown_keys or key in kept_keys else 0.0
            for key in all_keys
        }
        weights[method] = method_weights
        threshold_rows.append(
            {
                "diagnostic_label": DIAGNOSTIC_LABEL,
                "method": method,
                "keep_ratio": ratio,
                "oracle_known_anchor_count": len(known),
                "oracle_known_kept_count": keep_count,
                "oracle_unknown_anchor_count": len(unknown_keys),
                "propagation_enabled_anchor_count": int(sum(method_weights.values())),
                "threshold_h": (
                    float(kept[-1]["oracle_badness"]) if kept else math.nan
                ),
                "tie_break": "oracle_badness,frame_id,anchor_row,anchor_col",
            }
        )
    return weights, threshold_rows


def weights_for_frame(score_rows, method_weights):
    return np.asarray(
        [
            method_weights[(str(row["frame_id"]), int(row["anchor_row"]), int(row["anchor_col"]))]
            for row in score_rows
        ],
        dtype=np.float64,
    )


def safe_correlation(x, y, kind):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else math.nan


def build_oracle_summary(score_rows):
    known_rows = [row for row in score_rows if row["oracle_known"]]
    h = np.asarray([row["oracle_badness"] for row in known_rows], dtype=np.float64)
    discrepancy = np.asarray([row["abs_discrepancy_m"] for row in known_rows])
    log_discrepancy = np.asarray([row["abs_log_discrepancy"] for row in known_rows])
    total = len(score_rows)
    known = len(known_rows)
    return {
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "total_accepted_anchors": total,
        "oracle_known_anchors": known,
        "oracle_unknown_anchors": total - known,
        "unknown_ratio": (total - known) / total if total else math.nan,
        "mean_h": float(np.mean(h)) if h.size else math.nan,
        "std_h": float(np.std(h)) if h.size else math.nan,
        "median_h": _percentile(h, 50),
        "p10_h": _percentile(h, 10),
        "p25_h": _percentile(h, 25),
        "p50_h": _percentile(h, 50),
        "p75_h": _percentile(h, 75),
        "p90_h": _percentile(h, 90),
        "p95_h": _percentile(h, 95),
        "pearson_abs_discrepancy_m_vs_h": safe_correlation(discrepancy, h, "pearson"),
        "pearson_abs_log_discrepancy_vs_h": safe_correlation(log_discrepancy, h, "pearson"),
        "spearman_abs_discrepancy_m_vs_h": safe_correlation(discrepancy, h, "spearman"),
        "spearman_abs_log_discrepancy_vs_h": safe_correlation(log_discrepancy, h, "spearman"),
    }


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_method_csvs(metrics_dir, summary_rows, distance_rows):
    for method, _ in KEEP_RATIOS:
        method_summary = [row for row in summary_rows if row["method"] == method]
        method_distance = [row for row in distance_rows if row["method"] == method]
        write_csv(metrics_dir / f"{method}_summary.csv", method_summary)
        write_csv(metrics_dir / f"{method}_distance_summary.csv", method_distance)


def _float(row, key):
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def build_headroom_rows(summary_rows, frame_rows, threshold_rows):
    common = {
        row["method"]: row
        for row in summary_rows
        if row["area"] == "common_hidden_valid"
    }
    baseline = common["uniform"]
    thresholds = {row["method"]: row for row in threshold_rows}
    frames_by_method = defaultdict(list)
    for row in frame_rows:
        if row["area"] == "common_hidden_valid":
            frames_by_method[row["method"]].append(row)

    output = []
    metric_columns = {
        "MAE": "mae_weighted",
        "RMSE": "rmse_weighted",
        "median": "median_abs_mean",
        "P90": "p90_abs_mean",
        "P95": "p95_abs_mean",
        "P99": "p99_abs_mean",
    }
    for method, ratio in KEEP_RATIOS:
        row = common[method]
        uniform_by_frame = {
            item["name"]: _float(item, "mae") for item in frames_by_method["uniform"]
        }
        comparisons = [
            _float(item, "mae") - uniform_by_frame[item["name"]]
            for item in frames_by_method[method]
        ]
        improved = sum(value < 0 for value in comparisons)
        worsened = sum(value > 0 for value in comparisons)
        threshold = thresholds[method]
        output_row = {
            "diagnostic_label": DIAGNOSTIC_LABEL,
            "method": method,
            "keep_ratio": ratio,
            "oracle_known_anchor_count": threshold["oracle_known_anchor_count"],
            "propagation_enabled_anchor_count": threshold["propagation_enabled_anchor_count"],
            "frames_improved": improved,
            "frames_worsened": worsened,
            "frames_tied": len(comparisons) - improved - worsened,
            "improvement_ratio": improved / len(comparisons) if comparisons else math.nan,
        }
        for label, column in metric_columns.items():
            value = _float(row, column)
            baseline_value = _float(baseline, column)
            delta = value - baseline_value
            output_row[label] = value
            output_row[f"delta_{label}_vs_uniform"] = delta
            output_row[f"delta_{label}_pct"] = (
                100.0 * delta / baseline_value if baseline_value else math.nan
            )
        output.append(output_row)
    return output


def build_frame_diagnostics(frame_rows):
    common = [row for row in frame_rows if row["area"] == "common_hidden_valid"]
    uniform = {
        row["name"]: _float(row, "mae") for row in common if row["method"] == "uniform"
    }
    result = []
    for row in common:
        baseline = uniform[row["name"]]
        mae = _float(row, "mae")
        result.append(
            {
                "diagnostic_label": DIAGNOSTIC_LABEL,
                "frame_id": row["name"],
                "method": row["method"],
                "uniform_MAE": baseline,
                "variant_MAE": mae,
                "difference": mae - baseline,
                "improved": mae < baseline,
            }
        )
    return result


def load_config(config_path):
    with Path(config_path).open() as handle:
        config = yaml.safe_load(handle) or {}
    anchor_filter = dict(config.get("anchor_filter", {}))
    range_cfg = dict(config.get("range_gdc", {}))
    if anchor_filter.get("mode", "abs") != "abs":
        raise ValueError("Oracle diagnostic requires canonical anchor_filter.mode=abs")
    if float(anchor_filter.get("abs_error_thr", 2.0)) != 2.0:
        raise ValueError("Oracle diagnostic requires canonical abs_error_thr=2.0")
    force_policy = anchor_filter.get("force_policy", "accepted_only")
    if force_policy != "accepted_only":
        raise ValueError("Oracle diagnostic requires accepted_only force semantics")
    return {
        "method": range_cfg.get("method", "cg"),
        "range_min": float(range_cfg.get("range_min", 0.1)),
        "range_max": float(range_cfg.get("range_max", 80.0)),
        "anchor_reject": "abs",
        "log_ratio_thr": float(anchor_filter.get("log_ratio_thr", 0.4)),
        "abs_error_thr": 2.0,
        "lambda_anchor": float(range_cfg.get("lambda_anchor", 300.0)),
        "lambda_prior": float(range_cfg.get("lambda_prior", 0.1)),
        "lambda_smooth": float(range_cfg.get("lambda_smooth", 1.0)),
        "neighbor": range_cfg.get("neighbor", "angular_grid8"),
        "edge_spatial_mode": range_cfg.get("edge_spatial_mode", "angular"),
        "sigma_angular": float(range_cfg.get("sigma_angular", 0.01)),
        "sigma_tangent": float(range_cfg.get("sigma_tangent", 1.0)),
        "sigma_log_range": float(range_cfg.get("sigma_log_range", 0.3)),
        "max_log_range_diff": range_cfg.get("max_log_range_diff"),
        "delta_clip": range_cfg.get("delta_clip", 0.3),
        "anchor_force_policy": "accepted_only",
    }


def load_projection(base_root):
    gt_meta = base_root / "range" / "gt" / "meta" / "projection_meta.npz"
    anchor_meta = (
        base_root / "anchor" / "range_shared_canonical" / "meta" / "projection_meta.npz"
    )
    if not gt_meta.is_file() or not anchor_meta.is_file():
        raise FileNotFoundError("Required GT and canonical-anchor projection metadata are missing")
    with np.load(gt_meta, allow_pickle=True) as meta:
        vertical = meta["vertical_centers_deg"].astype(np.float64)
        azimuth = meta["azimuth_centers_deg"].astype(np.float64)
        azimuth_mode = str(meta["azimuth_mode"].item())
        height = int(meta["height"].item())
        width = int(meta["width"].item())
    with np.load(anchor_meta, allow_pickle=True) as meta:
        if "selected_rows" not in meta.files:
            raise ValueError("Canonical anchor projection metadata lacks selected_rows")
        source_rows = meta["selected_rows"].astype(np.int64)
    return {
        "path": gt_meta,
        "vertical_centers_deg": vertical,
        "azimuth_centers_deg": azimuth,
        "azimuth_mode": azimuth_mode,
        "shape": (height, width),
        "source_rows": source_rows,
    }


def range_kwargs(params, projection, target_weights=None):
    values = dict(params)
    values.update(
        {
            "vertical_centers_deg": projection["vertical_centers_deg"],
            "azimuth_centers_deg": projection["azimuth_centers_deg"],
            "azimuth_mode": projection["azimuth_mode"],
            "return_stats": True,
            "return_debug": True,
        }
    )
    if target_weights is not None:
        values["target_weights"] = target_weights
    return values


def _load_frame(maps, scene_id, expected_shape):
    arrays = [np.load(mapping[scene_id]).astype(np.float32) for mapping in maps]
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError(f"{scene_id}: input shape does not match projection {expected_shape}")
    return arrays


def _discover_inputs(base_root, scene_ids):
    paths = {
        "pred": base_root / "range" / "raw_sdn" / "G64_range",
        "anchor": base_root / "anchor" / "range_shared_canonical" / "G64_range",
        "gt": base_root / "range" / "gt" / "G64_range",
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Required existing {name} artifact directory is missing: {path}")
    maps = {name: npy_map(str(path)) for name, path in paths.items()}
    for name, mapping in maps.items():
        missing = [scene_id for scene_id in scene_ids if scene_id not in mapping]
        if missing:
            raise FileNotFoundError(f"{name}: missing {len(missing)} split frames, sample={missing[:5]}")
    return paths, maps


def _prepare_output(output_root, overwrite):
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output root is non-empty; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "oracle").mkdir()
    (output_root / "metrics").mkdir()
    for method, _ in KEEP_RATIOS:
        (output_root / "range" / method).mkdir(parents=True)


def _run_evaluator(paths, output_root, split_file, projection, params):
    metrics = output_root / "metrics"
    command = [
        sys.executable,
        str(REPO_ROOT / "range_gdc" / "evaluate_range_metrics.py"),
        "--split_file", str(split_file),
        "--gt_range_path", str(paths["gt"]),
        "--guide_range_path", str(paths["anchor"]),
        "--source_range_path", str(paths["pred"]),
        "--metrics_csv", str(metrics / "oracle_frame_metrics_full.csv"),
        "--summary_csv", str(metrics / "oracle_summary_full.csv"),
        "--projection_meta_path", str(projection["path"]),
        "--range_min", str(params["range_min"]),
        "--range_max", str(params["range_max"]),
        "--expected_height", str(projection["shape"][0]),
        "--expected_width", str(projection["shape"][1]),
        "--source_row_indices", *[str(row) for row in projection["source_rows"]],
        "--source_rows", str(len(projection["source_rows"])),
        "--target_h", str(projection["shape"][0]),
        "--enable_distance_bins",
        "--distance_bins", "0,10,20,30,40,50,60,70,80",
        "--distance_metrics_csv", str(metrics / "oracle_distance_metrics_full.csv"),
        "--distance_summary_csv", str(metrics / "oracle_distance_summary_full.csv"),
        "--raw_method", "uniform",
    ]
    for method, _ in KEEP_RATIOS:
        command.extend(["--method", f"{method}={output_root / 'range' / method}"])
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def run(args):
    started = time.perf_counter()
    validate_train1000_split(args.split_file)
    validate_diagnostic_roots(args.base_output_root, args.output_root)
    base_root = Path(args.base_output_root).resolve()
    output_root = Path(args.output_root).resolve()
    split_path = Path(args.split_file).resolve()
    params = load_config(args.config)
    projection = load_projection(base_root)
    scene_ids = read_split_scene_ids(str(split_path))
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive")
        scene_ids = scene_ids[: args.max_items]
    paths, maps = _discover_inputs(base_root, scene_ids)
    _prepare_output(output_root, args.overwrite)

    derived_split = output_root / "oracle" / TRAIN1000_SPLIT_NAME
    derived_split.write_text("".join(f"{scene_id}\n" for scene_id in scene_ids))
    config_record = {
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "base_output_root": str(base_root),
        "output_root": str(output_root),
        "source_split_file": str(split_path),
        "evaluated_split_file": str(derived_split),
        "frame_count": len(scene_ids),
        "source_rows": [int(row) for row in projection["source_rows"]],
        "range_gdc": params,
        "keep_ratios": dict(KEEP_RATIOS),
        "gt_usage": ["oracle anchor quality", "oracle experiment evaluation"],
    }
    (output_root / "oracle" / "oracle_manifest.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n"
    )

    print(f"{DIAGNOSTIC_LABEL}: scoring {len(scene_ids)} train1000 frame(s)")
    score_rows = []
    per_frame_scores = {}
    invariant_signatures = {}
    equivalence = {
        "frames_checked": 0,
        "corrected_range_max_abs_diff": 0.0,
        "accepted_mask_exact": True,
        "rejected_mask_exact": True,
        "graph_node_count_exact": True,
        "graph_edge_count_exact": True,
        "A_exact": True,
        "b_exact": True,
        "force_mask_exact": True,
    }
    for frame_index, scene_id in enumerate(scene_ids):
        pred, anchor, gt = _load_frame(
            (maps["pred"], maps["anchor"], maps["gt"]), scene_id, projection["shape"]
        )
        uniform, uniform_mask, _, debug = RangeROIGDC(
            pred, anchor, **range_kwargs(params, projection)
        )
        np.save(
            output_root / "range" / "uniform" / f"{scene_id}_G{pred.shape[0]}_corr_range.npy",
            uniform,
        )
        frame_scores = score_frame(
            scene_id, pred, anchor, gt, projection["source_rows"], debug, params
        )
        per_frame_scores[scene_id] = frame_scores
        score_rows.extend(frame_scores)
        invariant_signatures[scene_id] = graph_invariant_signature(
            debug, pred, anchor, params["range_min"], params["range_max"]
        )

        if frame_index < args.equivalence_frames:
            ones = np.ones(debug["target_node_indices"].shape, dtype=np.float64)
            keep100, keep100_mask, _, keep100_debug = RangeROIGDC(
                pred, anchor, **range_kwargs(params, projection, ones)
            )
            max_diff = float(np.max(np.abs(uniform.astype(np.float64) - keep100)))
            equivalence["frames_checked"] += 1
            equivalence["corrected_range_max_abs_diff"] = max(
                equivalence["corrected_range_max_abs_diff"], max_diff
            )
            equivalence["accepted_mask_exact"] &= np.array_equal(
                debug["target_mask"], keep100_debug["target_mask"]
            )
            rejected_uniform = (
                valid_range_mask(pred, params["range_min"], params["range_max"])
                & valid_range_mask(anchor, params["range_min"], params["range_max"])
                & ~debug["target_mask"]
            )
            rejected_keep100 = (
                valid_range_mask(pred, params["range_min"], params["range_max"])
                & valid_range_mask(anchor, params["range_min"], params["range_max"])
                & ~keep100_debug["target_mask"]
            )
            equivalence["rejected_mask_exact"] &= np.array_equal(
                rejected_uniform, rejected_keep100
            )
            equivalence["graph_node_count_exact"] &= (
                debug["node_rows"].size == keep100_debug["node_rows"].size
            )
            equivalence["graph_edge_count_exact"] &= (
                debug["edge_weight"].size == keep100_debug["edge_weight"].size
            )
            equivalence["A_exact"] &= (debug["A"] != keep100_debug["A"]).nnz == 0
            equivalence["b_exact"] &= np.array_equal(debug["b"], keep100_debug["b"])
            equivalence["force_mask_exact"] &= np.array_equal(
                debug["force_mask"], keep100_debug["force_mask"]
            )
            if not np.array_equal(uniform_mask, keep100_mask) or max_diff != 0.0:
                raise RuntimeError(f"{scene_id}: keep100 is not exactly uniform")

    method_weights, threshold_rows = rank_oracle_weights(score_rows)
    write_csv(output_root / "oracle" / "oracle_anchor_scores.csv", score_rows, SCORE_FIELDS)
    summary = build_oracle_summary(score_rows)
    summary.update({f"uniform_equivalence_{key}": value for key, value in equivalence.items()})
    summary.update(
        {
            f"{row['method']}_threshold_h": row["threshold_h"]
            for row in threshold_rows
        }
    )
    write_csv(output_root / "oracle" / "oracle_summary.csv", [summary])
    write_csv(output_root / "oracle" / "oracle_thresholds.csv", threshold_rows)

    for method, _ in KEEP_RATIOS[1:]:
        print(f"{DIAGNOSTIC_LABEL}: solving {method}")
        for scene_id in scene_ids:
            pred, anchor, _ = _load_frame(
                (maps["pred"], maps["anchor"], maps["gt"]), scene_id, projection["shape"]
            )
            weights = weights_for_frame(per_frame_scores[scene_id], method_weights[method])
            corrected, output_mask, _, debug = RangeROIGDC(
                pred, anchor, **range_kwargs(params, projection, weights)
            )
            signature = graph_invariant_signature(
                debug, pred, anchor, params["range_min"], params["range_max"]
            )
            assert_graph_invariants(invariant_signatures[scene_id], signature, scene_id, method)
            accepted = debug["target_mask"]
            if not np.array_equal(corrected[accepted], anchor[accepted]):
                raise RuntimeError(f"{scene_id} {method}: accepted exact forcing changed")
            if not np.array_equal(output_mask, valid_range_mask(corrected, params["range_min"], params["range_max"])):
                raise RuntimeError(f"{scene_id} {method}: output mask mismatch")
            np.save(
                output_root / "range" / method / f"{scene_id}_G{pred.shape[0]}_corr_range.npy",
                corrected,
            )

    _run_evaluator(paths, output_root, derived_split, projection, params)
    metrics_dir = output_root / "metrics"
    frame_rows = read_csv(metrics_dir / "oracle_frame_metrics_full.csv")
    summary_rows = read_csv(metrics_dir / "oracle_summary_full.csv")
    distance_rows = read_csv(metrics_dir / "oracle_distance_summary_full.csv")
    _write_method_csvs(metrics_dir, summary_rows, distance_rows)
    frame_diagnostics = build_frame_diagnostics(frame_rows)
    write_csv(output_root / "oracle" / "oracle_frame_diagnostics.csv", frame_diagnostics)
    headroom = build_headroom_rows(summary_rows, frame_rows, threshold_rows)
    best_methods = {
        metric: min(headroom, key=lambda row: float(row[metric]))["method"]
        for metric in ("MAE", "RMSE", "median", "P90", "P95", "P99")
    }
    for row in headroom:
        for metric, best_method in best_methods.items():
            row[f"best_method_by_{metric}"] = best_method
    write_csv(output_root / "oracle" / "oracle_headroom_summary.csv", headroom)

    elapsed = time.perf_counter() - started
    print(f"{DIAGNOSTIC_LABEL}: completed in {elapsed:.3f}s")
    print(f"Results: {output_root}")
    return {
        "frame_count": len(scene_ids),
        "elapsed_seconds": elapsed,
        "equivalence": equivalence,
        "summary": summary,
        "headroom": headroom,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "r64_pipeline.yaml")
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Train1000-prefix smoke only; does not authorize val/test Oracle runs.",
    )
    parser.add_argument("--equivalence-frames", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.equivalence_frames < 0:
        parser.error("--equivalence-frames must be nonnegative")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
