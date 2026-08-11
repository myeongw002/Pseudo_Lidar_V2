#!/usr/bin/env python3
"""Clean stage-based Range-GDC guide evaluation runner."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install requirements.txt first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from range_gdc.range_projection import find_input_npy  # noqa: E402


RANGE_H = 64
RANGE_W = 1024
VMIN_DEG = -24.9
VMAX_DEG = 2.0
AZIMUTH_MODE = "front_center"
INVALID_VALUE = 0.0
DEPTH_MIN = 0.1
DEPTH_MAX = 80.0
SELECTED_ROWS = [5, 7, 9, 11]

CORE_STAGES = [
    "sdn_depth",
    "canonical_shared_anchor",
    "canonical_shared_anchor_image_depth",
    "gt_range",
    "range_anchor_from_shared_anchor",
    "audit_shared_anchor_protocol",
    "sdn_depth_to_range",
    "original_gdc_naive",
    "original_gdc_naive_depth_to_range",
    "original_gdc_optimized",
    "original_gdc_optimized_depth_to_range",
    "range_gdc",
    "evaluate",
]

PREVIEW_STAGES = [
    "preview_sdn_depth",
    "preview_original_gdc_naive_depth",
    "preview_original_gdc_optimized_depth",
    "preview_gt_range",
    "preview_raw_sdn_range",
    "preview_original_gdc_naive_range",
    "preview_original_gdc_optimized_range",
    "preview_range_gdc_range",
]

POINTCLOUD_STAGES = [
    "raw_sdn_pointcloud",
    "original_gdc_naive_pointcloud",
    "original_gdc_optimized_pointcloud",
    "range_gdc_pointcloud",
]

@dataclass
class Stage:
    name: str
    inputs: list
    outputs: list
    clean_paths: list
    commands_fn: object
    validate_fn: object
    post_fn: object = None


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def split_values(values):
    result = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def read_split_ids(split_file):
    with open(split_file) as f:
        ids = [f"{int(line.strip()):06d}" for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No frame ids in split file: {split_file}")
    return ids


def load_yaml(path):
    if not path:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve(path):
    path = Path(os.path.expanduser(str(path)))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def add_arg(cmd, flag, value):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if value:
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        return
    cmd.extend([flag, str(value)])


def command_to_string(cmd):
    return " ".join(str(part) for part in cmd)


def path_is_under(path, root):
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (TypeError, ValueError):
        return False


def safe_delete(path, output_root):
    path = Path(path)
    if not path.exists():
        return
    if not path_is_under(path, output_root):
        raise ValueError(f"Refusing to delete path outside output_root: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def file_nonempty(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def frame_npy_exists(directory, scene_id):
    try:
        find_input_npy(str(directory), int(scene_id))
        return True
    except Exception:
        return False


def validate_file_count(directory, ids):
    missing = [scene_id for scene_id in ids if not frame_npy_exists(directory, scene_id)]
    if missing:
        raise ValueError(f"{directory}: missing {len(missing)}/{len(ids)} files, sample={missing[:5]}")


def load_frame_npy(directory, scene_id):
    return np.load(find_input_npy(str(directory), int(scene_id)))


def validate_range_dir(range_dir, ids, expected_shape=(RANGE_H, RANGE_W)):
    validate_file_count(range_dir, ids)
    for scene_id in ids:
        arr = load_frame_npy(range_dir, scene_id)
        if arr.shape != expected_shape:
            raise ValueError(f"{range_dir}/{scene_id}: expected {expected_shape}, got {arr.shape}")
        valid = np.isfinite(arr) & (arr > INVALID_VALUE)
        if int(valid.sum()) <= 0:
            raise ValueError(f"{range_dir}/{scene_id}: valid_count is zero")


def validate_depth_dir(depth_dir, ids):
    validate_file_count(depth_dir, ids)
    for scene_id in ids[: min(5, len(ids))]:
        arr = load_frame_npy(depth_dir, scene_id)
        valid = np.isfinite(arr) & (arr > 0)
        if int(valid.sum()) <= 0:
            raise ValueError(f"{depth_dir}/{scene_id}: valid depth count is zero")


def validate_image_source_index_dir(source_dir, ids):
    validate_file_count(source_dir, ids)
    for scene_id in ids[: min(5, len(ids))]:
        source = load_frame_npy(source_dir, scene_id)
        if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
            raise ValueError(f"{source_dir}/{scene_id}: expected 2D integer source-index map")
        if not np.any(source >= 0):
            raise ValueError(f"{source_dir}/{scene_id}: no valid GDC source winners")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shared_anchor_provenance(provenance_path, sparse_dir, source_index_dir, ids, projection, selected_rows):
    if not file_nonempty(provenance_path):
        raise ValueError(f"Missing shared-anchor provenance: {provenance_path}")
    with open(provenance_path) as f:
        provenance = json.load(f)

    errors = []
    if int(provenance.get("frame_count", -1)) != len(ids):
        errors.append(f"frame_count={provenance.get('frame_count')} expected={len(ids)}")
    definition = provenance.get("projection_definition", provenance)
    for name, expected in (("height", projection["height"]), ("width", projection["width"]),
                           ("vmin_deg", projection["vmin_deg"]), ("vmax_deg", projection["vmax_deg"]),
                           ("range_min", projection["depth_min"]), ("range_max", projection["depth_max"])):
        if not np.isclose(float(definition.get(name, np.nan)), float(expected)):
            errors.append(f"projection {name} mismatch")
    if definition.get("azimuth_mode") != projection["azimuth_mode"]:
        errors.append("projection azimuth_mode mismatch")
    if [int(v) for v in provenance.get("selected_rows", [])] != [int(v) for v in selected_rows]:
        errors.append("selected_rows mismatch")

    frame_by_id = {str(row.get("frame_id")): row for row in provenance.get("frames", [])}
    for scene_id in ids:
        row = frame_by_id.get(scene_id)
        if row is None:
            errors.append(f"missing provenance frame {scene_id}")
            continue
        output = Path(sparse_dir) / f"{scene_id}.bin"
        if not output.exists() or output.stat().st_size <= 0:
            errors.append(f"missing/empty shared anchor {scene_id}")
            continue
        expected_sha = str(row.get("sha256", ""))
        if not expected_sha or sha256_file(output) != expected_sha:
            errors.append(f"sha256 mismatch for {scene_id}")
        source_index = Path(source_index_dir) / f"{scene_id}.npy"
        if not source_index.exists() or np.load(source_index).shape != (projection["height"], projection["width"]):
            errors.append(f"missing/invalid source index for {scene_id}")
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError("Shared-anchor provenance validation failed: " + "; ".join(errors))


def validate_shared_canonical_range_anchor(range_dir, meta_dir, definition_path, provenance_path, ids, selected_rows, expected_shape):
    if not file_nonempty(definition_path):
        raise ValueError(f"Missing shared canonical anchor definition: {definition_path}")
    with open(definition_path) as handle:
        definition = json.load(handle)
    if definition.get("mode") != "shared_canonical" or definition.get("anchor_source") != "shared_canonical_pointcloud":
        raise ValueError("Range anchor is not declared as shared canonical")
    if Path(definition.get("source_provenance_path", "")).resolve() != Path(provenance_path).resolve():
        raise ValueError("Range anchor provenance path mismatch")
    if definition.get("source_manifest_sha256") != sha256_file(provenance_path):
        raise ValueError("Range anchor manifest checksum mismatch")
    rows = np.asarray(selected_rows, dtype=np.int32)
    for scene_id in ids:
        anchor = load_frame_npy(range_dir, scene_id)
        if anchor.shape != tuple(expected_shape):
            raise ValueError(f"{scene_id}: shared range anchor shape mismatch")
        valid = np.isfinite(anchor) & (anchor > INVALID_VALUE)
        if np.any(valid[np.setdiff1d(np.arange(expected_shape[0]), rows), :]):
            raise ValueError(f"{scene_id}: shared range anchor contains values outside selected rows")
        source = np.load(Path(meta_dir) / f"{scene_id}_source_index.npy")
        if source.shape != tuple(expected_shape) or np.any(valid & (source < 0)):
            raise ValueError(f"{scene_id}: RGC valid cells are not from the shared source")


def validate_pointcloud_dir(pc_dir, ids):
    missing = []
    empty = []
    for scene_id in ids:
        path = Path(pc_dir) / f"{scene_id}.bin"
        if not path.exists():
            missing.append(scene_id)
        elif path.stat().st_size <= 0:
            empty.append(scene_id)
    if missing:
        raise ValueError(f"{pc_dir}: missing {len(missing)}/{len(ids)} pointclouds, sample={missing[:5]}")
    if empty:
        raise ValueError(f"{pc_dir}: empty pointclouds, sample={empty[:5]}")


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def validate_shared_anchor_audit_outputs(
    audit_path, summary_path, provenance_path=None, sparse_dir=None, ids=None
):
    if not file_nonempty(audit_path) or not file_nonempty(summary_path):
        raise ValueError("missing shared-anchor audit outputs")
    summary = {row.get("metric"): row.get("value") for row in read_csv_rows(summary_path)}
    required_zero = (
        "shared_sparse_sha256_mismatch_count", "shared_point_count_mismatch_count",
        "rgc_source_index_grid_mismatch_count", "rgc_anchor_not_from_shared_count",
        "rgc_range_value_mismatch_count", "gdc_anchor_not_from_shared_count",
        "gdc_depth_value_mismatch_count",
    )
    missing = [name for name in required_zero if name not in summary]
    nonzero = [name for name in required_zero if name in summary and float(summary[name]) != 0.0]
    if missing or nonzero:
        raise ValueError(f"invalid shared-anchor audit summary: missing={missing}, nonzero={nonzero}")
    if provenance_path is not None and sparse_dir is not None and ids is not None:
        with open(provenance_path) as handle:
            provenance = json.load(handle)
        frame_rows = {str(row.get("frame_id")): row for row in provenance.get("frames", [])}
        changed = []
        for scene_id in ids:
            frame = frame_rows.get(str(scene_id))
            pointcloud = Path(sparse_dir) / f"{scene_id}.bin"
            if (frame is None or not pointcloud.is_file()
                    or frame.get("sha256") != sha256_file(pointcloud)):
                changed.append(str(scene_id))
        if changed:
            raise ValueError(f"shared PCD changed since manifest/audit: sample={changed[:5]}")


def validate_evaluation_outputs(paths, required_methods, leakage_enabled=True):
    for key in ("metrics_csv", "summary_csv"):
        if not file_nonempty(paths[key]):
            raise ValueError(f"Evaluation output is missing or empty: {paths[key]}")
    if paths.get("distance_eval_enabled", True):
        for key in ("distance_metrics_csv", "distance_summary_csv"):
            if not file_nonempty(paths[key]):
                raise ValueError(f"Distance evaluation output is missing or empty: {paths[key]}")
        validate_distance_summary(paths["distance_summary_csv"], required_methods)
    if not leakage_enabled:
        return
    for key in ("leakage_csv", "leakage_summary_csv"):
        if not file_nonempty(paths[key]):
            raise ValueError(f"Leakage output is missing or empty: {paths[key]}")
    rows = read_csv_rows(paths["leakage_summary_csv"])
    if not rows:
        raise ValueError("Leakage summary is empty")
    row = rows[0]
    if str(row.get("status", "")).upper() != "OK":
        raise ValueError(f"Leakage status is not OK: {row}")
    if int(float(row.get("frames_with_danger", 0))) != 0:
        raise ValueError(f"Leakage danger detected: {row}")
    if int(float(row.get("frames_with_warning", 0))) != 0:
        raise ValueError(f"Leakage warning detected: {row}")


def validate_distance_summary(path, required_methods):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Distance summary is empty: {path}")
    areas = {row.get("area") for row in rows}
    methods = {row.get("method") for row in rows}
    required_areas = {"hidden_rows_valid", "common_hidden_valid"}
    missing = []
    if not required_areas.issubset(areas):
        missing.append(f"areas={sorted(required_areas - areas)}")
    required_methods = set(required_methods)
    if not required_methods.issubset(methods):
        missing.append(f"methods={sorted(required_methods - methods)}")
    if missing:
        raise ValueError(f"Distance summary missing required rows: {', '.join(missing)}")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_context(args):
    cfg = load_yaml(args.config)
    output_root = resolve(args.output_root or cfg.get("output_root", "/data/kitti/pseudo_lidar_w1024"))
    kitti_root = resolve(args.kitti_root or cfg.get("kitti_root", "/data/kitti/kitti_object/testing"))
    split_file = resolve(args.split_file or cfg.get("split_file", "split/test.txt"))
    if args.data_tag:
        data_tag = args.data_tag
    elif args.split_file:
        data_tag = Path(split_file).stem
    else:
        data_tag = cfg.get("data_tag") or Path(split_file).stem
    threads = int(args.threads or cfg.get("threads", 4))

    anchor_cfg = dict(cfg.get("anchor", {}))
    if anchor_cfg.get("mode", "shared_canonical") != "shared_canonical":
        raise ValueError("anchor.mode must be shared_canonical")
    if "source_ptc_path" in anchor_cfg:
        raise ValueError(
            "anchor.source_ptc_path is no longer supported; "
            "LiDAR source is derived from kitti_root/velodyne"
        )
    anchor_rows = [int(v) for v in anchor_cfg.get("selected_rows", SELECTED_ROWS)]
    if not anchor_rows:
        raise ValueError("anchor.selected_rows must not be empty")

    range_anchor_cfg = dict(cfg.get("range_anchor", {}))
    if range_anchor_cfg.get("mode", "shared_canonical") != "shared_canonical":
        raise ValueError("range_anchor.mode must be shared_canonical")
    selected_rows = [
        int(v)
        for v in range_anchor_cfg.get("selected_rows", SELECTED_ROWS)
    ]
    if not selected_rows:
        raise ValueError("range_anchor.selected_rows must not be empty")
    if len(set(selected_rows)) != len(selected_rows):
        raise ValueError("range_anchor.selected_rows must not contain duplicates")
    if selected_rows != anchor_rows:
        raise ValueError("anchor.selected_rows and range_anchor.selected_rows must define the same canonical source rows")

    range_cfg = dict(cfg.get("range_gdc", {}))
    reliability_mode = (
        getattr(args, "anchor_reliability_mode", None)
        or range_cfg.get("anchor_reliability_mode", "uniform")
    )
    if reliability_mode not in {"uniform", "quadratic"}:
        raise ValueError("anchor_reliability_mode must be uniform or quadratic")
    range_cfg["anchor_reliability_mode"] = reliability_mode
    projection_cfg = dict(range_cfg.get("projection", {}))
    projection = {
        "height": int(projection_cfg.get("height", projection_cfg.get("range_h", RANGE_H))),
        "width": int(projection_cfg.get("width", projection_cfg.get("range_w", RANGE_W))),
        "vmin_deg": float(projection_cfg.get("vmin_deg", VMIN_DEG)),
        "vmax_deg": float(projection_cfg.get("vmax_deg", VMAX_DEG)),
        "azimuth_mode": projection_cfg.get("azimuth_mode", "full_360_front_centered"),
        "azimuth_min_deg": projection_cfg.get("azimuth_min_deg"),
        "azimuth_max_deg": projection_cfg.get("azimuth_max_deg"),
        "invalid_value": float(projection_cfg.get("invalid_value", INVALID_VALUE)),
        "depth_min": float(range_cfg.get("range_min", projection_cfg.get("depth_min", DEPTH_MIN))),
        "depth_max": float(range_cfg.get("range_max", projection_cfg.get("depth_max", DEPTH_MAX))),
        "threads": int(projection_cfg.get("threads", threads)),
        "image_fov_only": bool(projection_cfg.get("image_fov_only", True)),
    }
    if projection["height"] <= 0 or projection["width"] <= 0:
        raise ValueError("Projection height and width must be positive")

    original_cfg = dict(cfg.get("original_gdc", cfg.get("gdc", {})))
    anchor_filter = dict(cfg.get("anchor_filter", {}))
    evaluation_cfg = dict(cfg.get("evaluation", {}))

    paths = {
        "output_root": output_root,
        "config_dir": output_root / "config",
        "resolved_config": output_root / "config" / "resolved_config.yaml",
        "manifest": output_root / "config" / "manifest.json",
        "pipeline_summary": output_root / "config" / "pipeline_summary.json",
        "stage_timing": output_root / "config" / "stage_timing.csv",
        "command_log": output_root / "config" / "command_log.json",
        "kitti_root": kitti_root,
        "calib": kitti_root / "calib",
        "image": kitti_root / "image_2",
        "velodyne": kitti_root / "velodyne",
        "split_file": split_file,
        "sdn_depth": output_root / "sdn" / "depth_maps" / data_tag,
        "sdn_preview": output_root / "sdn" / "preview",
        "anchor_root": output_root / "anchor",
        "anchor_sparse_pc": output_root / "anchor" / "shared_canonical_pointcloud",
        "anchor_source_index": output_root / "anchor" / "shared_canonical_source_index",
        "anchor_provenance": output_root / "anchor" / "shared_canonical_pointcloud_provenance.json",
        "anchor_image_depth": output_root / "anchor" / "shared_canonical_image_depth",
        "anchor_image_source_index": output_root / "anchor" / "shared_canonical_image_source_index",
        "anchor_range_root": output_root / "anchor" / "range_shared_canonical",
        "anchor_range": output_root / "anchor" / "range_shared_canonical" / "G64_range",
        "anchor_mask": output_root / "anchor" / "range_shared_canonical" / "G64_mask",
        "anchor_meta": output_root / "anchor" / "range_shared_canonical" / "meta",
        "anchor_definition": output_root / "anchor" / "range_shared_canonical" / "meta" / "anchor_definition.json",
        "shared_anchor_audit": output_root / "anchor" / "shared_canonical_protocol_audit.csv",
        "shared_anchor_audit_summary": output_root / "anchor" / "shared_canonical_protocol_summary.csv",
        "gt_range_root": output_root / "range" / "gt",
        "gt_range": output_root / "range" / "gt" / "G64_range",
        "gt_mask": output_root / "range" / "gt" / "G64_mask",
        "gt_meta": output_root / "range" / "gt" / "meta",
        "gt_preview": output_root / "range" / "gt" / "preview",
        "raw_sdn_range_root": output_root / "range" / "raw_sdn",
        "raw_sdn_range": output_root / "range" / "raw_sdn" / "G64_range",
        "raw_sdn_mask": output_root / "range" / "raw_sdn" / "G64_mask",
        "raw_sdn_meta": output_root / "range" / "raw_sdn" / "meta",
        "raw_sdn_preview": output_root / "range" / "raw_sdn" / "preview",
        "original_gdc_naive_depth": output_root / "original_gdc" / "naive" / "corrected_depth",
        "original_gdc_naive_stats": output_root / "original_gdc" / "naive" / "stats" / "gdc_stats.csv",
        "original_gdc_naive_depth_preview": output_root / "original_gdc" / "naive" / "preview_depth",
        "original_gdc_naive_range_root": output_root / "range" / "original_gdc_naive",
        "original_gdc_naive_range": output_root / "range" / "original_gdc_naive" / "G64_range",
        "original_gdc_naive_mask": output_root / "range" / "original_gdc_naive" / "G64_mask",
        "original_gdc_naive_meta": output_root / "range" / "original_gdc_naive" / "meta",
        "original_gdc_naive_range_preview": output_root / "range" / "original_gdc_naive" / "preview",
        "original_gdc_optimized_depth": output_root / "original_gdc" / "optimized" / "corrected_depth",
        "original_gdc_optimized_stats": output_root / "original_gdc" / "optimized" / "stats" / "gdc_stats.csv",
        "original_gdc_optimized_depth_preview": output_root / "original_gdc" / "optimized" / "preview_depth",
        "original_gdc_optimized_range_root": output_root / "range" / "original_gdc_optimized",
        "original_gdc_optimized_range": output_root / "range" / "original_gdc_optimized" / "G64_range",
        "original_gdc_optimized_mask": output_root / "range" / "original_gdc_optimized" / "G64_mask",
        "original_gdc_optimized_meta": output_root / "range" / "original_gdc_optimized" / "meta",
        "original_gdc_optimized_range_preview": output_root / "range" / "original_gdc_optimized" / "preview",
        "range_gdc_range_root": output_root / "range" / "range_gdc",
        "range_gdc_range": output_root / "range" / "range_gdc" / "G64_range",
        "range_gdc_mask": output_root / "range" / "range_gdc" / "G64_mask",
        "range_gdc_meta": output_root / "range" / "range_gdc" / "meta",
        "range_gdc_preview": output_root / "range" / "range_gdc" / "preview",
        "pc_raw_sdn": output_root / "pointcloud" / "raw_sdn_64ch",
        "pc_original_gdc_naive": output_root / "pointcloud" / "original_gdc_naive_64ch",
        "pc_original_gdc_optimized": output_root / "pointcloud" / "original_gdc_optimized_64ch",
        "pc_range_gdc": output_root / "pointcloud" / "range_gdc_64ch",
        "metrics_dir": output_root / "metrics",
        "metrics_csv": output_root / "metrics" / "guide_r64_metrics.csv",
        "summary_csv": output_root / "metrics" / "guide_r64_summary.csv",
        "distance_metrics_csv": output_root / "metrics" / "guide_r64_distance_metrics.csv",
        "distance_summary_csv": output_root / "metrics" / "guide_r64_distance_summary.csv",
        "distance_eval_enabled": not args.no_distance_eval,
        "leakage_csv": output_root / "metrics" / "range_gdc_leakage_check.csv",
        "leakage_summary_csv": output_root / "metrics" / "range_gdc_leakage_summary.csv",
        "projection_meta": output_root / "range" / "gt" / "meta" / "projection_meta.npz",
    }

    resolved = {
        "output_root": str(output_root),
        "kitti_root": str(kitti_root),
        "split_file": str(split_file),
        "data_tag": data_tag,
        "threads": threads,
        "projection": projection,
        "anchor": {**anchor_cfg, "selected_rows": anchor_rows},
        "range_anchor": {**range_anchor_cfg, "selected_rows": selected_rows},
        "anchor_filter": anchor_filter,
        "original_gdc": original_cfg,
        "range_gdc": range_cfg,
        "evaluation": evaluation_cfg,
        "sdn": {
            "config": str(resolve(args.sdn_config or cfg.get("sdn_config", "src/configs/sdn_kitti_train.config"))),
            "checkpoint": str(resolve(args.sdn_checkpoint or cfg.get("sdn_checkpoint", "/data/kitti/pseudo_lidar/sdn_kitti_train_set/checkpoint.pth.tar"))),
            "save_path": str(output_root / "sdn"),
        },
    }
    return {
        "config": cfg,
        "resolved": resolved,
        "paths": paths,
        "ids": read_split_ids(split_file),
        "threads": threads,
        "data_tag": data_tag,
        "projection": projection,
        "selected_rows": selected_rows,
        "anchor_cfg": anchor_cfg,
        "range_anchor_cfg": range_anchor_cfg,
        "anchor_filter": anchor_filter,
        "original_gdc": original_cfg,
        "range_gdc": range_cfg,
        "evaluation": evaluation_cfg,
    }


def range_projection_cmd(mode, input_path, output_root, ctx, image_fov_only=None):
    p = ctx["paths"]
    projection = ctx["projection"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "range_gdc" / "range_projection.py"),
        mode,
        "--input_path", str(input_path),
        "--output_path", str(output_root),
        "--calib_path", str(p["calib"]),
        "--split_file", str(p["split_file"]),
        "--threads", str(projection["threads"]),
        "--height", str(projection["height"]),
        "--width", str(projection["width"]),
        "--vmin_deg", str(projection["vmin_deg"]),
        "--vmax_deg", str(projection["vmax_deg"]),
        "--azimuth_mode", str(projection["azimuth_mode"]),
        "--depth_min", str(projection["depth_min"]),
        "--depth_max", str(projection["depth_max"]),
        "--invalid_value", str(projection["invalid_value"]),
        "--anchor_rows", *[str(row) for row in ctx["selected_rows"]],
        "--meta_path", str(p["projection_meta"]),
        "--stats_csv", str(Path(output_root) / "meta" / "projection_stats.csv"),
    ]
    if projection.get("azimuth_min_deg") is not None:
        add_arg(cmd, "--azimuth_min_deg", projection["azimuth_min_deg"])
    if projection.get("azimuth_max_deg") is not None:
        add_arg(cmd, "--azimuth_max_deg", projection["azimuth_max_deg"])
    if mode == "ptc-to-range":
        cmd.extend(["--image_path", str(p["image"])])
        if image_fov_only is None:
            image_fov_only = projection["image_fov_only"]
        cmd.append("--image_fov_only" if image_fov_only else "--no-image_fov_only")
    return cmd


def preview_cmd(input_path, output_path, ctx):
    projection = ctx["projection"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "preview_npy.py"),
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--split_file", str(ctx["paths"]["split_file"]),
        "--vmin", str(projection["depth_min"]),
        "--vmax", str(projection["depth_max"]),
    ]
    if ctx["args"].preview_max_items is not None:
        cmd.extend(["--max_items", str(ctx["args"].preview_max_items)])
    return cmd


def pointcloud_cmd(range_path, output_path, ctx):
    projection = ctx["projection"]
    return [
        sys.executable,
        str(REPO_ROOT / "range_gdc" / "range_to_pointcloud.py"),
        "--range_path", str(range_path),
        "--output_path", str(output_path),
        "--split_file", str(ctx["paths"]["split_file"]),
        "--projection_meta_path", str(ctx["paths"]["projection_meta"]),
        "--height", str(projection["height"]),
        "--width", str(projection["width"]),
        "--vmin_deg", str(projection["vmin_deg"]),
        "--vmax_deg", str(projection["vmax_deg"]),
        "--range_min", str(projection["depth_min"]),
        "--range_max", str(projection["depth_max"]),
    ]


def copy_projection_meta_to(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    src = path.parents[1] / "gt" / "meta" / "projection_meta.npz"
    if src.exists():
        shutil.copy2(src, path / "projection_meta.npz")


def build_stages(ctx):
    p = ctx["paths"]
    c = ctx["config"]
    anchor = ctx["anchor_cfg"]
    anchor_filter = ctx["anchor_filter"]
    original = ctx["original_gdc"]
    rgdc = ctx["range_gdc"]
    evaluation = ctx["evaluation"]
    projection = ctx["projection"]
    expected_shape = (projection["height"], projection["width"])

    def sdn_commands():
        return [[
            sys.executable,
            str(REPO_ROOT / "src" / "main.py"),
            "-c", ctx["resolved"]["sdn"]["config"],
            "--resume", ctx["resolved"]["sdn"]["checkpoint"],
            "--datapath", str(p["kitti_root"]),
            "--data_list", str(p["split_file"]),
            "--generate_depth_map",
            "--data_tag", ctx["data_tag"],
            "--save_path", str(p["output_root"] / "sdn"),
            "--workers", str(c.get("sdn_workers", ctx["threads"])),
            "--bval", str(c.get("bval", 1)),
        ]]

    def shared_anchor_pointcloud_commands():
        cmd = [
            sys.executable, str(REPO_ROOT / "tools" / "create_shared_canonical_anchor.py"),
            "--velodyne-dir", str(p["velodyne"]),
            "--split-file", str(p["split_file"]),
            "--output-pointcloud-dir", str(p["anchor_sparse_pc"]),
            "--output-source-index-dir", str(p["anchor_source_index"]),
            "--provenance-json", str(p["anchor_provenance"]),
            "--height", str(projection["height"]), "--width", str(projection["width"]),
            "--vmin-deg", str(projection["vmin_deg"]), "--vmax-deg", str(projection["vmax_deg"]),
            "--azimuth-mode", str(projection["azimuth_mode"]),
            "--range-min", str(projection["depth_min"]), "--range-max", str(projection["depth_max"]),
            "--invalid-value", str(projection["invalid_value"]),
            "--selected-rows", *[str(row) for row in ctx["selected_rows"]],
        ]
        add_arg(cmd, "--azimuth-min-deg", projection.get("azimuth_min_deg"))
        add_arg(cmd, "--azimuth-max-deg", projection.get("azimuth_max_deg"))
        return [cmd]

    def shared_anchor_image_depth_commands():
        return [[
            sys.executable,
            str(REPO_ROOT / "gdc" / "ptc2depthmap.py"),
            "--input_path", str(p["anchor_sparse_pc"]),
            "--output_path", str(p["anchor_image_depth"]),
            "--calib_path", str(p["calib"]),
            "--image_path", str(p["image"]),
            "--split_file", str(p["split_file"]),
            "--threads", str(anchor.get("threads", ctx["threads"])),
            "--collision-policy", "nearest_positive",
            "--provenance-json", str(p["anchor_provenance"]),
            "--source-index-input-path", str(p["anchor_source_index"]),
            "--source-index-output-path", str(p["anchor_image_source_index"]),
            "--selected-rows", *[str(row) for row in ctx["selected_rows"]],
        ]]


    def range_anchor_commands():
        return [[
            sys.executable,
            str(REPO_ROOT / "range_gdc" / "build_range_anchor.py"),
            "--output_range_path", str(p["anchor_range"]),
            "--output_mask_path", str(p["anchor_mask"]),
            "--split_file", str(p["split_file"]),
            "--selected_rows", *[str(row) for row in ctx["selected_rows"]],
            "--invalid_value", str(projection["invalid_value"]),
            "--expected_height", str(projection["height"]),
            "--expected_width", str(projection["width"]),
            "--projection_meta_path", str(p["projection_meta"]),
            "--meta_dir", str(p["anchor_meta"]),
            "--threads", str(ctx["range_anchor_cfg"].get("threads", ctx["threads"])),
            "--source-index-dir", str(p["anchor_source_index"]),
            "--shared-pointcloud-dir", str(p["anchor_sparse_pc"]),
            "--source-provenance-path", str(p["anchor_provenance"]),
        ]]

    def shared_anchor_audit_commands():
        return [[
            sys.executable, str(REPO_ROOT / "tools" / "audit_shared_anchor_protocol.py"),
            "--output-root", str(p["output_root"]), "--split-file", str(p["split_file"]),
            "--calib-dir", str(p["calib"]), "--image-dir", str(p["image"]),
            "--output-csv", str(p["shared_anchor_audit"]),
            "--summary-csv", str(p["shared_anchor_audit_summary"]),
        ]]

    def original_gdc_command(variant):
        if variant not in {"naive", "optimized"}:
            raise ValueError(f"Unknown Original GDC variant: {variant}")
        output = p[f"original_gdc_{variant}_depth"]
        stats = p[f"original_gdc_{variant}_stats"]
        cmd = [
            sys.executable,
            str(REPO_ROOT / "gdc" / "main_batch.py"),
            "--input_path", str(p["sdn_depth"]),
            "--calib_path", str(p["calib"]),
            "--gt_depthmap_path", str(p["anchor_image_depth"]),
            "--output_path", str(output),
            "--split_file", str(p["split_file"]),
            "--threads", str(original.get("threads", ctx["threads"])),
            "--stats_csv", str(stats),
            "--overwrite",
            "--anchor_reject", str(anchor_filter.get("mode", "abs")),
            "--abs_error_thr", str(anchor_filter.get("abs_error_thr", 2.0)),
            "--log_ratio_thr", str(anchor_filter.get("log_ratio_thr", 0.4)),
            "--anchor_force_policy", str(anchor_filter.get("force_policy", "accepted_only")),
        ]
        add_arg(cmd, "--k", original.get("k", 10))
        add_arg(cmd, "--recon_tol", original.get("recon_tol", 0.0005))
        add_arg(cmd, "--method", original.get("method", "cg"))
        add_arg(cmd, "--consider_range", original.get("consider_range", [-0.1, 3.0]))
        if variant == "naive":
            cmd.append("--disable_subsample")
        else:
            add_arg(cmd, "--subsample_strategy", original.get("subsample_strategy", "deterministic"))
            add_arg(cmd, "--subsample_seed", original.get("subsample_seed", 0))
            add_arg(cmd, "--subsample_output", original.get("subsample_output", "preserve"))
        return [cmd]

    def range_gdc_commands():
        cmd = [
            sys.executable,
            str(REPO_ROOT / "range_gdc" / "range_main_batch.py"),
            "--pred_path", str(p["raw_sdn_range"]),
            "--anchor_path", str(p["anchor_range"]),
            "--output_path", str(p["range_gdc_range"]),
            "--mask_output_path", str(p["range_gdc_mask"]),
            "--projection_meta_path", str(p["projection_meta"]),
            "--meta_dir", str(p["range_gdc_meta"]),
            "--stats_csv", str(p["range_gdc_meta"] / "range_gdc_stats.csv"),
            "--threads", str(rgdc.get("threads", ctx["threads"])),
            "--overwrite",
            "--anchor_reject", str(anchor_filter.get("mode", "abs")),
            "--abs_error_thr", str(anchor_filter.get("abs_error_thr", 2.0)),
            "--log_ratio_thr", str(anchor_filter.get("log_ratio_thr", 0.4)),
            "--anchor_force_policy", str(anchor_filter.get("force_policy", rgdc.get("anchor_force_policy", "accepted_only"))),
        ]
        for name in (
            "method", "range_min", "range_max", "anchor_reliability_mode",
            "lambda_anchor", "lambda_prior",
            "lambda_smooth", "neighbor", "edge_spatial_mode", "sigma_angular",
            "sigma_tangent", "sigma_log_range", "max_log_range_diff", "delta_clip",
        ):
            if name in rgdc:
                add_arg(cmd, "--" + name, rgdc[name])
        return [cmd]

    method_paths = []
    if evaluation.get("include_raw", True):
        method_paths.append(("sdn_raw", p["raw_sdn_range"]))
    include_original = evaluation.get("include_original_gdc", True)
    if include_original and original.get("run_naive", True):
        method_paths.append(("original_gdc_naive", p["original_gdc_naive_range"]))
    if include_original and original.get("run_optimized", True):
        method_paths.append(("original_gdc_optimized", p["original_gdc_optimized_range"]))
    if evaluation.get("include_range_gdc", True):
        method_paths.append(("range_gdc", p["range_gdc_range"]))
    required_methods = [name for name, _ in method_paths]
    leakage_enabled = bool(evaluation.get("enable_leakage_check", True))

    def evaluate_commands():
        cmd = [
            sys.executable,
            str(REPO_ROOT / "range_gdc" / "evaluate_range_metrics.py"),
            "--split_file", str(p["split_file"]),
            "--gt_range_path", str(p["gt_range"]),
            "--guide_range_path", str(p["anchor_range"]),
            "--source_range_path", str(p["raw_sdn_range"]),
            "--metrics_csv", str(p["metrics_csv"]),
            "--summary_csv", str(p["summary_csv"]),
            "--projection_meta_path", str(p["projection_meta"]),
            "--range_min", str(evaluation.get("range_min", projection["depth_min"])),
            "--range_max", str(evaluation.get("range_max", projection["depth_max"])),
            "--expected_height", str(projection["height"]),
            "--expected_width", str(projection["width"]),
            "--source_row_indices", *[str(row) for row in ctx["selected_rows"]],
            "--source_rows", str(evaluation.get("source_rows", len(ctx["selected_rows"]))),
            "--target_h", str(projection["height"]),
        ]
        for name, method_path in method_paths:
            cmd.extend(["--method", f"{name}={method_path}"])
        if leakage_enabled:
            cmd.extend([
                "--enable_leakage_check",
                "--anchor_range_path", str(p["anchor_range"]),
                "--leakage_method", "range_gdc",
                "--leakage_output_csv", str(p["leakage_csv"]),
                "--leakage_summary_csv", str(p["leakage_summary_csv"]),
            ])
            if evaluation.get("allow_leakage", False):
                cmd.append("--allow_leakage")
        if not ctx["args"].no_distance_eval:
            bins = evaluation.get("distance_bins", [0, 10, 20, 30, 40, 50, 60, 70, 80])
            cmd.extend([
                "--enable_distance_bins",
                "--distance_bins", ",".join(str(v) for v in bins),
                "--distance_metrics_csv", str(p["distance_metrics_csv"]),
                "--distance_summary_csv", str(p["distance_summary_csv"]),
            ])
        return [cmd]

    stages = [
        Stage("sdn_depth", [p["split_file"]], [p["sdn_depth"]], [p["sdn_depth"]], sdn_commands, lambda ids: validate_depth_dir(p["sdn_depth"], ids)),
        Stage(
            "canonical_shared_anchor",
            [p["velodyne"]],
            [p["anchor_sparse_pc"], p["anchor_source_index"], p["anchor_provenance"]],
            [p["anchor_sparse_pc"], p["anchor_source_index"], p["anchor_provenance"]],
            shared_anchor_pointcloud_commands,
            lambda ids: validate_shared_anchor_provenance(p["anchor_provenance"], p["anchor_sparse_pc"], p["anchor_source_index"], ids, projection, ctx["selected_rows"]),
        ),
        Stage(
            "canonical_shared_anchor_image_depth",
            [p["anchor_sparse_pc"], p["anchor_source_index"], p["anchor_provenance"]],
            [p["anchor_image_depth"], p["anchor_image_source_index"]],
            [p["anchor_image_depth"], p["anchor_image_source_index"]],
            shared_anchor_image_depth_commands,
            lambda ids: (
                validate_depth_dir(p["anchor_image_depth"], ids),
                validate_image_source_index_dir(p["anchor_image_source_index"], ids),
            ),
        ),
        Stage("gt_range", [p["velodyne"]], [p["gt_range"], p["gt_mask"], p["gt_meta"]], [p["gt_range_root"]], lambda: [range_projection_cmd("ptc-to-range", p["velodyne"], p["gt_range_root"], ctx, image_fov_only=not ctx["args"].full_lidar_gt)], lambda ids: validate_range_dir(p["gt_range"], ids, expected_shape)),
        Stage(
            "range_anchor_from_shared_anchor",
            [p["anchor_sparse_pc"], p["anchor_source_index"], p["anchor_provenance"], p["projection_meta"]],
            [p["anchor_range"], p["anchor_mask"], p["anchor_meta"], p["anchor_definition"]],
            [p["anchor_range_root"]],
            range_anchor_commands,
            lambda ids: validate_shared_canonical_range_anchor(
                p["anchor_range"],
                p["anchor_meta"],
                p["anchor_definition"],
                p["anchor_provenance"],
                ids,
                ctx["selected_rows"],
                expected_shape,
            ),
        ),
        Stage(
            "audit_shared_anchor_protocol",
            [p["anchor_sparse_pc"], p["anchor_source_index"], p["anchor_provenance"], p["anchor_image_depth"], p["anchor_range"]],
            [p["shared_anchor_audit"], p["shared_anchor_audit_summary"]],
            [p["shared_anchor_audit"], p["shared_anchor_audit_summary"]],
            shared_anchor_audit_commands,
            lambda ids: validate_shared_anchor_audit_outputs(
                p["shared_anchor_audit"], p["shared_anchor_audit_summary"],
                p["anchor_provenance"], p["anchor_sparse_pc"], ids,
            ),
        ),
        Stage("sdn_depth_to_range", [p["sdn_depth"]], [p["raw_sdn_range"], p["raw_sdn_mask"], p["raw_sdn_meta"]], [p["raw_sdn_range_root"]], lambda: [range_projection_cmd("depth-to-range", p["sdn_depth"], p["raw_sdn_range_root"], ctx)], lambda ids: validate_range_dir(p["raw_sdn_range"], ids, expected_shape)),
        Stage("original_gdc_naive", [p["sdn_depth"], p["anchor_image_depth"]], [p["original_gdc_naive_depth"], p["original_gdc_naive_stats"]], [p["original_gdc_naive_depth"], p["original_gdc_naive_stats"].parent], lambda: original_gdc_command("naive"), lambda ids: (validate_depth_dir(p["original_gdc_naive_depth"], ids), file_nonempty(p["original_gdc_naive_stats"]) or (_ for _ in ()).throw(ValueError("missing naive GDC stats")))),
        Stage("original_gdc_naive_depth_to_range", [p["original_gdc_naive_depth"]], [p["original_gdc_naive_range"], p["original_gdc_naive_mask"], p["original_gdc_naive_meta"]], [p["original_gdc_naive_range_root"]], lambda: [range_projection_cmd("depth-to-range", p["original_gdc_naive_depth"], p["original_gdc_naive_range_root"], ctx)], lambda ids: validate_range_dir(p["original_gdc_naive_range"], ids, expected_shape)),
        Stage("original_gdc_optimized", [p["sdn_depth"], p["anchor_image_depth"]], [p["original_gdc_optimized_depth"], p["original_gdc_optimized_stats"]], [p["original_gdc_optimized_depth"], p["original_gdc_optimized_stats"].parent], lambda: original_gdc_command("optimized"), lambda ids: (validate_depth_dir(p["original_gdc_optimized_depth"], ids), file_nonempty(p["original_gdc_optimized_stats"]) or (_ for _ in ()).throw(ValueError("missing optimized GDC stats")))),
        Stage("original_gdc_optimized_depth_to_range", [p["original_gdc_optimized_depth"]], [p["original_gdc_optimized_range"], p["original_gdc_optimized_mask"], p["original_gdc_optimized_meta"]], [p["original_gdc_optimized_range_root"]], lambda: [range_projection_cmd("depth-to-range", p["original_gdc_optimized_depth"], p["original_gdc_optimized_range_root"], ctx)], lambda ids: validate_range_dir(p["original_gdc_optimized_range"], ids, expected_shape)),
        Stage("range_gdc", [p["raw_sdn_range"], p["anchor_range"]], [p["range_gdc_range"], p["range_gdc_mask"], p["range_gdc_meta"]], [p["range_gdc_range_root"]], range_gdc_commands, lambda ids: validate_range_dir(p["range_gdc_range"], ids, expected_shape), lambda: copy_projection_meta_to(p["range_gdc_meta"])),
    ]

    evaluation_inputs = [p["gt_range"], p["anchor_range"], *[path for _, path in method_paths]]
    evaluation_outputs = [p["metrics_csv"], p["summary_csv"]]
    if leakage_enabled:
        evaluation_outputs.extend([p["leakage_csv"], p["leakage_summary_csv"]])
    if not ctx["args"].no_distance_eval:
        evaluation_outputs.extend([p["distance_metrics_csv"], p["distance_summary_csv"]])
    stages.append(Stage("evaluate", evaluation_inputs, evaluation_outputs, [p["metrics_dir"]], evaluate_commands, lambda ids: validate_evaluation_outputs(p, required_methods, leakage_enabled)))

    preview_specs = [
        ("preview_sdn_depth", p["sdn_depth"], p["sdn_preview"]),
        ("preview_original_gdc_naive_depth", p["original_gdc_naive_depth"], p["original_gdc_naive_depth_preview"]),
        ("preview_original_gdc_optimized_depth", p["original_gdc_optimized_depth"], p["original_gdc_optimized_depth_preview"]),
        ("preview_gt_range", p["gt_range"], p["gt_preview"]),
        ("preview_raw_sdn_range", p["raw_sdn_range"], p["raw_sdn_preview"]),
        ("preview_original_gdc_naive_range", p["original_gdc_naive_range"], p["original_gdc_naive_range_preview"]),
        ("preview_original_gdc_optimized_range", p["original_gdc_optimized_range"], p["original_gdc_optimized_range_preview"]),
        ("preview_range_gdc_range", p["range_gdc_range"], p["range_gdc_preview"]),
    ]
    for name, input_path, output_path in preview_specs:
        stages.append(Stage(name, [input_path], [output_path], [output_path], lambda input_path=input_path, output_path=output_path: [preview_cmd(input_path, output_path, ctx)], lambda ids, output_path=output_path: None if Path(output_path).exists() else (_ for _ in ()).throw(ValueError(f"missing preview dir {output_path}"))))

    pc_specs = [
        ("raw_sdn_pointcloud", p["raw_sdn_range"], p["pc_raw_sdn"]),
        ("original_gdc_naive_pointcloud", p["original_gdc_naive_range"], p["pc_original_gdc_naive"]),
        ("original_gdc_optimized_pointcloud", p["original_gdc_optimized_range"], p["pc_original_gdc_optimized"]),
        ("range_gdc_pointcloud", p["range_gdc_range"], p["pc_range_gdc"]),
    ]
    for name, input_path, output_path in pc_specs:
        stages.append(Stage(name, [input_path], [output_path], [output_path], lambda input_path=input_path, output_path=output_path: [pointcloud_cmd(input_path, output_path, ctx)], lambda ids, output_path=output_path: validate_pointcloud_dir(output_path, ids)))
    return {stage.name: stage for stage in stages}


def ordered_stage_names(args, ctx):
    original = ctx["original_gdc"]
    order = [
        "sdn_depth",
        "canonical_shared_anchor",
        "canonical_shared_anchor_image_depth",
        "gt_range",
        "range_anchor_from_shared_anchor",
        "audit_shared_anchor_protocol",
        "sdn_depth_to_range",
    ]
    if original.get("run_naive", True):
        order.extend(["original_gdc_naive", "original_gdc_naive_depth_to_range"])
    if original.get("run_optimized", True):
        order.extend(["original_gdc_optimized", "original_gdc_optimized_depth_to_range"])
    order.extend(["range_gdc", "evaluate"])

    if args.make_preview:
        preview_after = {
            "sdn_depth": "preview_sdn_depth",
            "original_gdc_naive": "preview_original_gdc_naive_depth",
            "original_gdc_optimized": "preview_original_gdc_optimized_depth",
            "gt_range": "preview_gt_range",
            "sdn_depth_to_range": "preview_raw_sdn_range",
            "original_gdc_naive_depth_to_range": "preview_original_gdc_naive_range",
            "original_gdc_optimized_depth_to_range": "preview_original_gdc_optimized_range",
            "range_gdc": "preview_range_gdc_range",
        }
        expanded = []
        for stage in order:
            expanded.append(stage)
            if stage in preview_after:
                expanded.append(preview_after[stage])
        order = expanded
    if args.export_pointcloud:
        insert_at = order.index("evaluate")
        pc_stages = ["raw_sdn_pointcloud"]
        if original.get("run_naive", True):
            pc_stages.append("original_gdc_naive_pointcloud")
        if original.get("run_optimized", True):
            pc_stages.append("original_gdc_optimized_pointcloud")
        pc_stages.append("range_gdc_pointcloud")
        for stage in pc_stages:
            order.insert(insert_at, stage)
            insert_at += 1
    return order


def select_stages(args, order):
    selected = order[:]
    if args.only_stage:
        selected = [args.only_stage]
    if args.stages:
        selected = split_values([args.stages])
    if args.from_stage:
        start = order.index(args.from_stage)
        selected = [name for name in selected if order.index(name) >= start]
    if args.to_stage:
        end = order.index(args.to_stage)
        selected = [name for name in selected if order.index(name) <= end]
    skip = set(split_values(args.skip_stage))
    return [name for name in selected if name not in skip]


def stage_complete(stage, ids):
    try:
        stage.validate_fn(ids)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def run_stage(stage, ids, force, dry_run, output_root):
    complete, reason = stage_complete(stage, ids)
    action = "skip" if complete else "run"
    if force:
        action = "force_run"
    commands = stage.commands_fn()

    print(f"\n[{stage.name}]")
    print(f"inputs : {', '.join(str(p) for p in stage.inputs)}")
    print(f"outputs: {', '.join(str(p) for p in stage.outputs)}")
    if reason and action != "skip":
        print(f"status : incomplete ({reason})")
    for cmd in commands:
        print("command:", command_to_string(cmd))
    print(f"action : {'dry-run ' + action if dry_run else action}")

    started = now_iso()
    perf = time.perf_counter()
    result = {
        "stage": stage.name,
        "action": action,
        "started_at": started,
        "input_paths": [str(p) for p in stage.inputs],
        "output_paths": [str(p) for p in stage.outputs],
        "commands": commands,
        "complete_before": complete,
        "validation_before": reason,
    }
    try:
        if not dry_run and action != "skip":
            for path in stage.clean_paths:
                safe_delete(path, output_root)
            for cmd in commands:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            if stage.post_fn is not None:
                stage.post_fn()
            stage.validate_fn(ids)
        result["ended_at"] = now_iso()
        result["duration_sec"] = round(time.perf_counter() - perf, 6)
        return result
    except Exception as exc:
        result["action"] = "failed"
        result["ended_at"] = now_iso()
        result["duration_sec"] = round(time.perf_counter() - perf, 6)
        result["error"] = str(exc)
        raise
    finally:
        if "ended_at" not in result:
            result["ended_at"] = now_iso()
            result["duration_sec"] = round(time.perf_counter() - perf, 6)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "action",
        "started_at",
        "ended_at",
        "duration_sec",
        "input_paths",
        "output_paths",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row.get("stage"),
                    "action": row.get("action"),
                    "started_at": row.get("started_at"),
                    "ended_at": row.get("ended_at"),
                    "duration_sec": row.get("duration_sec"),
                    "input_paths": json.dumps(row.get("input_paths", [])),
                    "output_paths": json.dumps(row.get("output_paths", [])),
                }
            )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_run_artifacts(ctx, selected, results, dry_run):
    p = ctx["paths"]
    if dry_run:
        return
    p["config_dir"].mkdir(parents=True, exist_ok=True)
    with open(p["resolved_config"], "w") as f:
        yaml.safe_dump(ctx["resolved"], f, sort_keys=False)
    write_csv(p["stage_timing"], results)
    write_json(p["command_log"], results)
    manifest = {
        "run_root": str(p["output_root"]),
        "split_file": str(p["split_file"]),
        "frame_count": len(ctx["ids"]),
        "selected_rows": ctx["selected_rows"],
        "range_h": ctx["projection"]["height"],
        "range_w": ctx["projection"]["width"],
        "projection": ctx["projection"],
        "anchor_mode": "shared_canonical_pointcloud",
        "anchor_provenance": str(p["anchor_provenance"]),
        "stage_list": selected,
        "git_commit": git_commit(),
        "created_artifacts": {key: str(value) for key, value in p.items() if path_is_under(value, p["output_root"])},
    }
    write_json(p["manifest"], manifest)
    write_json(p["pipeline_summary"], {"manifest": manifest, "stages": results})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/r64_pipeline.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--kitti-root", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument(
        "--anchor-reliability-mode",
        choices=["uniform", "quadratic"],
        default=None,
    )
    parser.add_argument("--data-tag", default=None)
    parser.add_argument("--sdn-config", default=None)
    parser.add_argument("--sdn-checkpoint", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--to-stage", default=None)
    parser.add_argument("--only-stage", default=None)
    parser.add_argument("--skip-stage", action="append", default=[])
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--force-from", default=None)
    parser.add_argument("--clean-stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--make-preview",
        dest="make_preview",
        action="store_true",
        default=True,
        help="Generate previews. This is enabled by default.",
    )
    parser.add_argument(
        "--no-preview",
        dest="make_preview",
        action="store_false",
        help="Disable preview generation.",
    )
    parser.add_argument("--preview-max-items", type=int, default=None)
    parser.add_argument(
        "--export-pointcloud",
        dest="export_pointcloud",
        action="store_true",
        default=True,
        help="Export 64ch pointclouds. This is enabled by default.",
    )
    parser.add_argument(
        "--no-export-pointcloud",
        dest="export_pointcloud",
        action="store_false",
        help="Disable 64ch pointcloud export.",
    )
    parser.add_argument(
        "--no-distance-eval",
        action="store_true",
        help="Disable GT-range-bin distance evaluation in the evaluate stage.",
    )
    parser.add_argument(
        "--full-lidar-gt",
        action="store_true",
        help="Use full 360-degree LiDAR GT range by adding --no-image_fov_only to ptc-to-range. Default is camera/image-FOV GT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ctx = build_context(args)
    ctx["args"] = args
    stages = build_stages(ctx)
    order = ordered_stage_names(args, ctx)
    selected = select_stages(args, order)
    for name in selected:
        if name not in stages:
            raise ValueError(f"Unknown stage: {name}")

    if args.clean_stage:
        name = args.clean_stage
        if name not in stages:
            raise ValueError(f"Unknown stage: {name}")
        print(f"Cleaning stage: {name}")
        for path in stages[name].clean_paths:
            print(f"delete: {path}")
            if not args.dry_run:
                safe_delete(path, ctx["paths"]["output_root"])
        return

    force_set = set(split_values(args.force_stage))
    if args.force:
        force_set.update(selected)
    if args.force_from:
        start = order.index(args.force_from)
        force_set.update(name for name in order[start:] if name in selected)

    print("Clean Range-GDC experiment")
    print(f"output_root: {ctx['paths']['output_root']}")
    print(f"frames     : {len(ctx['ids'])}")
    print(f"stages     : {', '.join(selected)}")
    print(f"dry_run    : {args.dry_run}")

    results = []
    try:
        for name in selected:
            result = run_stage(
                stages[name],
                ctx["ids"],
                force=(name in force_set or not args.resume),
                dry_run=args.dry_run,
                output_root=ctx["paths"]["output_root"],
            )
            results.append(result)
    finally:
        write_run_artifacts(ctx, selected, results, args.dry_run)

    print("\nDone.")
    if not args.dry_run:
        print(f"manifest     : {ctx['paths']['manifest']}")
        print(f"stage_timing : {ctx['paths']['stage_timing']}")
        print(f"command_log  : {ctx['paths']['command_log']}")


if __name__ == "__main__":
    main()
