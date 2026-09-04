#!/usr/bin/env python3
"""Paired single-frame latency benchmark for image-space GDC and RGC.

The benchmark loads each frame before starting the timer, runs both correction
functions in one Python process, and randomizes their order for every
frame/repeat pair. The primary result is the distribution of per-frame median
latencies, which avoids treating repeated measurements as independent samples.

Default paths follow the ``pseudo_lidar_final_trainval`` layout used by the RGC
experiments. GDC receives predicted image depth, sparse image-depth anchors,
and KITTI calibration. RGC receives the corresponding predicted range grid and
sparse range-grid anchors. Thus, this measures each method's correction core in
its native representation; common SDN inference and output file I/O are not
included.

Example
-------
python tools/benchmark_gdc_rgc_runtime.py \
    --data-root /media/myungw00/2TB_SSD/kitti \
    --frames 100 \
    --warmup-frames 5 \
    --repeats 3 \
    --output-dir /media/myungw00/2TB_SSD/kitti/runtime_gdc_rgc

For the paper, use exactly the RGC configuration that produced the reported
accuracy. The defaults below follow the effective settings of the checked-in
``r64_pipeline.yaml`` and runner. Explicit flags are provided for the newer
metric-residual manuscript formulation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("gdc-rgc-runtime")


@dataclass(frozen=True)
class RunRecord:
    frame_id: str
    repeat: int
    execution_order: int
    method: str
    elapsed_ms: float
    internal_ms: float
    node_count: int


@dataclass(frozen=True)
class FrameResult:
    frame_id: str
    gdc_ms: float
    rgc_ms: float
    rgc_minus_gdc_ms: float
    speedup_gdc_over_rgc: float


@dataclass(frozen=True)
class MethodSummary:
    method: str
    frames: int
    repeats: int
    mean_ms: float
    std_ms: float
    median_ms: float
    p05_ms: float
    p95_ms: float
    mean_fps: float
    mean_ci95_low_ms: float
    mean_ci95_high_ms: float


def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GDC and RGC correction latency on paired KITTI frames, "
            "excluding file loading and output saving."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="KITTI workspace root, e.g. /media/.../kitti",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for CSV/JSON/plots"
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Frame list; defaults to openpcdet_variants/raw/ImageSets/val.txt",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Uniformly sampled measured frames; 0 uses all available frames",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Uniformly sampled frames run once per method before measurement",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP thread limit applied equally to both methods",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Paired-frame bootstrap samples for mean-latency 95%% intervals",
    )

    pipeline_root_help = "Override a default pseudo_lidar_final_trainval path"
    parser.add_argument("--gdc-pred-depth-dir", type=Path, help=pipeline_root_help)
    parser.add_argument("--gdc-anchor-depth-dir", type=Path, help=pipeline_root_help)
    parser.add_argument("--calib-dir", type=Path, help=pipeline_root_help)
    parser.add_argument("--rgc-pred-range-dir", type=Path, help=pipeline_root_help)
    parser.add_argument("--rgc-anchor-range-dir", type=Path, help=pipeline_root_help)
    parser.add_argument("--projection-meta", type=Path, help=pipeline_root_help)

    shared = parser.add_argument_group("shared anchor policy")
    shared.add_argument(
        "--anchor-reject", choices=("none", "abs", "log_ratio"), default="abs"
    )
    shared.add_argument("--abs-error-thr", type=float, default=2.0)
    shared.add_argument("--log-ratio-thr", type=float, default=0.4)
    shared.add_argument(
        "--anchor-force-policy",
        choices=("accepted_only", "all_valid", "none"),
        default="accepted_only",
    )

    gdc = parser.add_argument_group("GDC parameters")
    gdc.add_argument("--gdc-k", type=int, default=10)
    gdc.add_argument("--gdc-method", choices=("cg", "gmres"), default="cg")
    gdc.add_argument("--gdc-recon-tol", type=float, default=5e-4)
    gdc.add_argument("--gdc-w-tol", type=float, default=1e-5)
    gdc.add_argument(
        "--gdc-consider-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        default=(-0.1, 3.0),
    )
    gdc.add_argument(
        "--gdc-subsample",
        action="store_true",
        help="Enable GDC 0.1-m grid subsampling; off matches the naive baseline",
    )
    gdc.add_argument(
        "--gdc-subsample-strategy",
        choices=("legacy_random", "seeded_random", "deterministic"),
        default="deterministic",
    )

    rgc = parser.add_argument_group("RGC parameters")
    rgc.add_argument("--rgc-method", choices=("cg", "spsolve"), default="cg")
    rgc.add_argument("--range-min", type=float, default=0.1)
    rgc.add_argument("--range-max", type=float, default=80.0)
    rgc.add_argument("--lambda-anchor", type=float, default=300.0)
    rgc.add_argument("--lambda-prior", type=float, default=0.1)
    rgc.add_argument("--lambda-smooth", type=float, default=1.0)
    rgc.add_argument(
        "--neighbor", choices=("angular_grid4", "angular_grid8"), default="angular_grid8"
    )
    rgc.add_argument(
        "--edge-spatial-mode", choices=("angular", "tangent"), default="angular"
    )
    rgc.add_argument(
        "--edge-range-mode", choices=("log_gaussian", "uniform"), default="log_gaussian"
    )
    rgc.add_argument(
        "--residual-domain",
        choices=("linear", "log"),
        default="log",
        help="Use the value matching the accuracy experiment",
    )
    rgc.add_argument("--sigma-angular", type=float, default=0.01)
    rgc.add_argument("--sigma-tangent", type=float, default=1.0)
    rgc.add_argument("--sigma-log-range", type=float, default=0.3)
    rgc.add_argument("--max-log-range-diff", type=float)
    rgc.add_argument(
        "--delta-clip",
        type=float,
        default=0.3,
        help="Residual clipping used by the checked-in pipeline (default: 0.3)",
    )
    rgc.add_argument(
        "--disable-delta-clip",
        action="store_const",
        const=None,
        dest="delta_clip",
        help="Disable residual clipping to match the current manuscript statement",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_root = args.data_root / "pseudo_lidar_final_trainval"
    return {
        "split_file": (
            args.data_root
            / "openpcdet_variants"
            / "raw"
            / "ImageSets"
            / "val.txt"
        ),
        "gdc_pred_depth_dir": output_root / "sdn" / "depth_maps" / "trainval_final",
        "gdc_anchor_depth_dir": output_root / "anchor" / "shared_canonical_image_depth",
        "calib_dir": args.data_root / "kitti_object_original" / "training" / "calib",
        "rgc_pred_range_dir": output_root / "range" / "raw_sdn" / "G64_range",
        "rgc_anchor_range_dir": output_root / "anchor" / "range_shared_canonical" / "G64_range",
        "projection_meta": output_root / "range" / "gt" / "meta" / "projection_meta.npz",
    }


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    defaults = default_paths(args)
    return {
        name: getattr(args, name) if getattr(args, name) is not None else default
        for name, default in defaults.items()
    }


def validate_args(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.frames < 0:
        raise ValueError("--frames must be nonnegative")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be nonnegative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.gdc_k <= 0:
        raise ValueError("--gdc-k must be positive")
    if args.gdc_recon_tol <= 0 or args.gdc_w_tol <= 0:
        raise ValueError("GDC tolerances must be positive")
    if args.gdc_consider_range[0] >= args.gdc_consider_range[1]:
        raise ValueError("--gdc-consider-range must satisfy MIN_DEG < MAX_DEG")
    if not 0 <= args.range_min < args.range_max:
        raise ValueError("range limits must satisfy 0 <= MIN < MAX")
    if min(args.lambda_anchor, args.lambda_prior, args.lambda_smooth) < 0:
        raise ValueError("RGC lambda values must be nonnegative")
    if min(args.sigma_angular, args.sigma_tangent, args.sigma_log_range) <= 0:
        raise ValueError("RGC sigma values must be positive")
    if args.max_log_range_diff is not None and args.max_log_range_diff <= 0:
        raise ValueError("--max-log-range-diff must be positive")
    if args.delta_clip is not None and args.delta_clip <= 0:
        raise ValueError("--delta-clip must be positive")
    if args.abs_error_thr < 0 or args.log_ratio_thr < 0:
        raise ValueError("anchor rejection thresholds must be nonnegative")

    for name, path in paths.items():
        if name in {"split_file", "projection_meta"}:
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        elif not path.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not args.repo_root.is_dir():
        raise FileNotFoundError(f"repo root does not exist: {args.repo_root}")


def configure_thread_environment(thread_count: int) -> None:
    value = str(thread_count)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value


def import_algorithms(repo_root: Path) -> tuple[Callable[..., Any], Callable[..., Any], type]:
    gdc_dir = repo_root / "gdc"
    rgc_dir = repo_root / "range_gdc"
    for directory in (gdc_dir, rgc_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"method directory does not exist: {directory}")

    sys.path.insert(0, str(gdc_dir))
    try:
        gdc_module = importlib.import_module("gdc")
        calibration_module = importlib.import_module("data_utils.kitti_util")
    except Exception as exc:
        raise RuntimeError(f"failed to import GDC implementation: {exc}") from exc

    sys.path.insert(0, str(rgc_dir))
    try:
        rgc_module = importlib.import_module("range_gdc")
    except Exception as exc:
        raise RuntimeError(f"failed to import RGC implementation: {exc}") from exc

    return gdc_module.GDC, rgc_module.RangeROIGDC, calibration_module.Calibration


def read_frame_ids(path: Path) -> list[str]:
    frame_ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        token = line.split("#", maxsplit=1)[0].strip()
        if not token:
            continue
        token = token.split()[0]
        if not token.isdigit():
            raise ValueError(f"invalid frame ID at {path}:{line_number}: {token!r}")
        frame_ids.append(f"{int(token):06d}")
    frame_ids = list(dict.fromkeys(frame_ids))
    if not frame_ids:
        raise RuntimeError(f"split contains no frame IDs: {path}")
    return frame_ids


def find_array(directory: Path, frame_id: str) -> Path:
    candidates = (
        directory / f"{frame_id}.npy",
        directory / f"{frame_id}_G64_range.npy",
        directory / f"{frame_id}_G64_corr_range.npy",
        directory / f"{frame_id}_range.npy",
    )
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(directory.glob(f"{frame_id}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"ambiguous arrays for frame {frame_id} in {directory}: "
            + ", ".join(path.name for path in matches)
        )
    raise FileNotFoundError(f"no array for frame {frame_id} in {directory}")


def available_frame_ids(frame_ids: Sequence[str], paths: dict[str, Path]) -> list[str]:
    available: list[str] = []
    missing: list[str] = []
    array_directories = (
        paths["gdc_pred_depth_dir"],
        paths["gdc_anchor_depth_dir"],
        paths["rgc_pred_range_dir"],
        paths["rgc_anchor_range_dir"],
    )
    for frame_id in frame_ids:
        try:
            for directory in array_directories:
                find_array(directory, frame_id)
            calibration = paths["calib_dir"] / f"{frame_id}.txt"
            if not calibration.is_file():
                raise FileNotFoundError(calibration)
        except (FileNotFoundError, RuntimeError):
            missing.append(frame_id)
        else:
            available.append(frame_id)
    if missing:
        LOGGER.warning(
            "%d split frames are missing at least one benchmark input; sample=%s",
            len(missing),
            missing[:5],
        )
    if not available:
        raise RuntimeError("no frame has every GDC/RGC benchmark input")
    return available


def uniform_sample(frame_ids: Sequence[str], count: int) -> list[str]:
    if count == 0 or count >= len(frame_ids):
        return list(frame_ids)
    positions = np.linspace(0, len(frame_ids) - 1, count, dtype=np.int64)
    return [frame_ids[int(position)] for position in positions]


def scalar_from_npz(meta: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in meta.files:
        return default
    value = meta[key]
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def load_projection_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as meta:
        result = {
            "vertical_centers_deg": (
                np.asarray(meta["vertical_centers_deg"], dtype=np.float64)
                if "vertical_centers_deg" in meta.files
                else None
            ),
            "azimuth_centers_deg": (
                np.asarray(meta["azimuth_centers_deg"], dtype=np.float64)
                if "azimuth_centers_deg" in meta.files
                else None
            ),
            "azimuth_mode": str(
                scalar_from_npz(meta, "azimuth_mode", "full_360_front_centered")
            ),
            "height": scalar_from_npz(meta, "height"),
            "width": scalar_from_npz(meta, "width"),
        }
    return result


def load_array(path: Path, label: str) -> np.ndarray:
    try:
        array = np.asarray(np.load(path, allow_pickle=False)).squeeze()
    except Exception as exc:
        raise RuntimeError(f"failed to load {label} from {path}: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{label} must be 2-D after squeeze, got {array.shape}")
    return array.astype(np.float32, copy=False)


def thread_limit_context(thread_count: int):
    try:
        module = importlib.import_module("threadpoolctl")
    except ImportError:
        return nullcontext()
    return module.threadpool_limits(limits=thread_count)


def timed_call(function: Callable[[], Any], thread_count: int) -> tuple[Any, float]:
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with thread_limit_context(thread_count):
            started_ns = time.perf_counter_ns()
            result = function()
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    finally:
        if gc_was_enabled:
            gc.enable()
    return result, elapsed_ms


def build_method_calls(
    args: argparse.Namespace,
    frame_id: str,
    paths: dict[str, Path],
    projection: dict[str, Any],
    gdc_function: Callable[..., Any],
    rgc_function: Callable[..., Any],
    calibration_class: type,
) -> dict[str, Callable[[], Any]]:
    pred_depth = load_array(
        find_array(paths["gdc_pred_depth_dir"], frame_id), "GDC predicted depth"
    )
    anchor_depth = load_array(
        find_array(paths["gdc_anchor_depth_dir"], frame_id), "GDC anchor depth"
    )
    pred_range = load_array(
        find_array(paths["rgc_pred_range_dir"], frame_id), "RGC predicted range"
    )
    anchor_range = load_array(
        find_array(paths["rgc_anchor_range_dir"], frame_id), "RGC anchor range"
    )
    if pred_depth.shape != anchor_depth.shape:
        raise ValueError(
            f"{frame_id}: GDC depth shape mismatch {pred_depth.shape} vs {anchor_depth.shape}"
        )
    if pred_range.shape != anchor_range.shape:
        raise ValueError(
            f"{frame_id}: RGC range shape mismatch {pred_range.shape} vs {anchor_range.shape}"
        )
    expected_shape = (
        None
        if projection["height"] is None or projection["width"] is None
        else (int(projection["height"]), int(projection["width"]))
    )
    if expected_shape is not None and pred_range.shape != expected_shape:
        raise ValueError(
            f"{frame_id}: RGC shape {pred_range.shape} does not match metadata {expected_shape}"
        )
    calibration = calibration_class(str(paths["calib_dir"] / f"{frame_id}.txt"))

    def run_gdc() -> Any:
        return gdc_function(
            pred_depth,
            anchor_depth,
            calibration,
            k=args.gdc_k,
            W_tol=args.gdc_w_tol,
            recon_tol=args.gdc_recon_tol,
            method=args.gdc_method,
            consider_range=tuple(args.gdc_consider_range),
            subsample=args.gdc_subsample,
            subsample_strategy=args.gdc_subsample_strategy,
            subsample_seed=args.seed + int(frame_id),
            subsample_output="preserve",
            anchor_reject=args.anchor_reject,
            abs_error_thr=args.abs_error_thr,
            log_ratio_thr=args.log_ratio_thr,
            anchor_force_policy=args.anchor_force_policy,
            return_stats=True,
        )

    def run_rgc() -> Any:
        return rgc_function(
            pred_range,
            anchor_range,
            vertical_centers_deg=projection["vertical_centers_deg"],
            azimuth_centers_deg=projection["azimuth_centers_deg"],
            azimuth_mode=projection["azimuth_mode"],
            method=args.rgc_method,
            range_min=args.range_min,
            range_max=args.range_max,
            anchor_reject=args.anchor_reject,
            log_ratio_thr=args.log_ratio_thr,
            abs_error_thr=args.abs_error_thr,
            lambda_anchor=args.lambda_anchor,
            lambda_prior=args.lambda_prior,
            lambda_smooth=args.lambda_smooth,
            neighbor=args.neighbor,
            edge_spatial_mode=args.edge_spatial_mode,
            sigma_angular=args.sigma_angular,
            sigma_tangent=args.sigma_tangent,
            sigma_log_range=args.sigma_log_range,
            max_log_range_diff=args.max_log_range_diff,
            edge_range_mode=args.edge_range_mode,
            residual_domain=args.residual_domain,
            delta_clip=args.delta_clip,
            anchor_force_policy=args.anchor_force_policy,
            return_stats=True,
        )

    return {"GDC": run_gdc, "RGC": run_rgc}


def unpack_result(method: str, result: Any) -> tuple[float, int]:
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"{method} did not return output and statistics")
    stats = result[-1]
    if not isinstance(stats, dict):
        raise RuntimeError(f"{method} statistics are not a dictionary")
    if method == "GDC":
        internal_ms = math.nan
        node_count = int(stats.get("correction_node_count", 0))
    else:
        internal_ms = 1000.0 * float(stats.get("t_total_correction", math.nan))
        node_count = int(stats.get("N_nodes", 0))
    return internal_ms, node_count


def run_warmup(
    args: argparse.Namespace,
    frame_ids: Sequence[str],
    paths: dict[str, Path],
    projection: dict[str, Any],
    algorithms: tuple[Callable[..., Any], Callable[..., Any], type],
) -> None:
    if not frame_ids:
        return
    LOGGER.info("warming up both methods on %d frames", len(frame_ids))
    gdc_function, rgc_function, calibration_class = algorithms
    for frame_id in frame_ids:
        calls = build_method_calls(
            args,
            frame_id,
            paths,
            projection,
            gdc_function,
            rgc_function,
            calibration_class,
        )
        for method in ("GDC", "RGC"):
            result, _ = timed_call(calls[method], args.cpu_threads)
            unpack_result(method, result)
            del result


def run_benchmark(
    args: argparse.Namespace,
    frame_ids: Sequence[str],
    paths: dict[str, Path],
    projection: dict[str, Any],
    algorithms: tuple[Callable[..., Any], Callable[..., Any], type],
) -> list[RunRecord]:
    rng = np.random.default_rng(args.seed)
    gdc_function, rgc_function, calibration_class = algorithms
    records: list[RunRecord] = []
    total_pairs = len(frame_ids) * args.repeats
    completed_pairs = 0
    for frame_index, frame_id in enumerate(frame_ids, 1):
        calls = build_method_calls(
            args,
            frame_id,
            paths,
            projection,
            gdc_function,
            rgc_function,
            calibration_class,
        )
        for repeat in range(args.repeats):
            order = ["GDC", "RGC"]
            rng.shuffle(order)
            for execution_order, method in enumerate(order, 1):
                result, elapsed_ms = timed_call(calls[method], args.cpu_threads)
                internal_ms, node_count = unpack_result(method, result)
                records.append(
                    RunRecord(
                        frame_id=frame_id,
                        repeat=repeat,
                        execution_order=execution_order,
                        method=method,
                        elapsed_ms=elapsed_ms,
                        internal_ms=internal_ms,
                        node_count=node_count,
                    )
                )
                del result
            completed_pairs += 1
            if completed_pairs % 10 == 0 or completed_pairs == total_pairs:
                LOGGER.info(
                    "completed %d/%d frame-repeat pairs", completed_pairs, total_pairs
                )
        LOGGER.debug("completed frame %d/%d: %s", frame_index, len(frame_ids), frame_id)
    return records


def write_dataclass_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    LOGGER.info("saved %s", path)


def frame_results(records: Sequence[RunRecord]) -> list[FrameResult]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        grouped.setdefault((record.frame_id, record.method), []).append(record.elapsed_ms)
    frame_ids = sorted({record.frame_id for record in records})
    results: list[FrameResult] = []
    for frame_id in frame_ids:
        gdc_values = grouped.get((frame_id, "GDC"), [])
        rgc_values = grouped.get((frame_id, "RGC"), [])
        if not gdc_values or not rgc_values:
            raise RuntimeError(f"missing paired measurements for frame {frame_id}")
        gdc_ms = float(np.median(gdc_values))
        rgc_ms = float(np.median(rgc_values))
        results.append(
            FrameResult(
                frame_id=frame_id,
                gdc_ms=gdc_ms,
                rgc_ms=rgc_ms,
                rgc_minus_gdc_ms=rgc_ms - gdc_ms,
                speedup_gdc_over_rgc=gdc_ms / rgc_ms,
            )
        )
    return results


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    count = values.size
    means = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(samples, 1000))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, count, size=(stop - start, count))
        means[start:stop] = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def summarize_methods(
    results: Sequence[FrameResult], repeats: int, bootstrap_samples: int, seed: int
) -> list[MethodSummary]:
    rng = np.random.default_rng(seed)
    summaries: list[MethodSummary] = []
    for method, field in (("GDC", "gdc_ms"), ("RGC", "rgc_ms")):
        values = np.asarray([getattr(item, field) for item in results], dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(values, bootstrap_samples, rng)
        mean_ms = float(np.mean(values))
        summaries.append(
            MethodSummary(
                method=method,
                frames=values.size,
                repeats=repeats,
                mean_ms=mean_ms,
                std_ms=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                median_ms=float(np.median(values)),
                p05_ms=float(np.percentile(values, 5)),
                p95_ms=float(np.percentile(values, 95)),
                mean_fps=1000.0 / mean_ms,
                mean_ci95_low_ms=ci_low,
                mean_ci95_high_ms=ci_high,
            )
        )
    return summaries


def save_summary_json(
    path: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    summaries: Sequence[MethodSummary],
    results: Sequence[FrameResult],
) -> None:
    gdc = next(item for item in summaries if item.method == "GDC")
    rgc = next(item for item in summaries if item.method == "RGC")
    paired_differences = np.asarray(
        [item.rgc_minus_gdc_ms for item in results], dtype=np.float64
    )
    paired_speedups = np.asarray(
        [item.speedup_gdc_over_rgc for item in results], dtype=np.float64
    )
    payload = {
        "benchmark_scope": (
            "correction function only; input loading, output saving, SDN inference, "
            "and depth/range projection excluded"
        ),
        "hardware": {
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "cpu_threads_limited_to": args.cpu_threads,
        },
        "configuration": {
            "measured_frames": len(results),
            "warmup_frames": args.warmup_frames,
            "repeats": args.repeats,
            "seed": args.seed,
            "gdc": {
                "k": args.gdc_k,
                "method": args.gdc_method,
                "recon_tol": args.gdc_recon_tol,
                "subsample": args.gdc_subsample,
                "consider_range_deg": list(args.gdc_consider_range),
            },
            "rgc": {
                "method": args.rgc_method,
                "neighbor": args.neighbor,
                "residual_domain": args.residual_domain,
                "edge_spatial_mode": args.edge_spatial_mode,
                "edge_range_mode": args.edge_range_mode,
                "lambda_anchor": args.lambda_anchor,
                "lambda_prior": args.lambda_prior,
                "lambda_smooth": args.lambda_smooth,
                "sigma_angular": args.sigma_angular,
                "sigma_log_range": args.sigma_log_range,
                "delta_clip": args.delta_clip,
            },
            "shared_anchor_policy": {
                "reject": args.anchor_reject,
                "abs_error_thr": args.abs_error_thr,
                "log_ratio_thr": args.log_ratio_thr,
                "force_policy": args.anchor_force_policy,
            },
        },
        "paths": {name: str(value) for name, value in paths.items()},
        "method_summaries": [asdict(item) for item in summaries],
        "paired_comparison": {
            "mean_rgc_minus_gdc_ms": float(np.mean(paired_differences)),
            "median_rgc_minus_gdc_ms": float(np.median(paired_differences)),
            "mean_gdc_over_rgc_speedup": float(np.mean(paired_speedups)),
            "median_gdc_over_rgc_speedup": float(np.median(paired_speedups)),
            "rgc_faster_frame_fraction": float(np.mean(paired_differences < 0)),
            "mean_latency_ratio_gdc_over_rgc": gdc.mean_ms / rgc.mean_ms,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("saved %s", path)


def save_runtime_plot(
    path_stem: Path,
    results: Sequence[FrameResult],
    dpi: int = 300,
) -> None:
    gdc = np.asarray([item.gdc_ms for item in results], dtype=np.float64)
    rgc = np.asarray([item.rgc_ms for item in results], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55))

    boxes = axes[0].boxplot(
        (gdc, rgc),
        tick_labels=("GDC", "RGC"),
        showfliers=False,
        patch_artist=True,
        widths=0.52,
    )
    for patch, color in zip(boxes["boxes"], ("#E69F00", "#0072B2"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_ylabel("Correction latency per frame (ms)")
    axes[0].set_title("(a) Per-frame median latency", loc="left")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(gdc, rgc, s=14, alpha=0.60, color="#4C72B0", edgecolors="none")
    lower = max(0.0, float(min(np.min(gdc), np.min(rgc))) * 0.95)
    upper = float(max(np.max(gdc), np.max(rgc))) * 1.05
    axes[1].plot((lower, upper), (lower, upper), "--", color="0.35", linewidth=1.0)
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("GDC latency (ms)")
    axes[1].set_ylabel("RGC latency (ms)")
    axes[1].set_title("(b) Paired frames", loc="left")
    axes[1].grid(alpha=0.25)
    axes[1].set_aspect("equal", adjustable="box")

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = path_stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        LOGGER.info("saved %s", path)
    plt.close(fig)


def print_summary(summaries: Sequence[MethodSummary], results: Sequence[FrameResult]) -> None:
    for item in summaries:
        LOGGER.info(
            "%s: mean %.3f ms (95%% CI %.3f–%.3f), median %.3f ms, P95 %.3f ms",
            item.method,
            item.mean_ms,
            item.mean_ci95_low_ms,
            item.mean_ci95_high_ms,
            item.median_ms,
            item.p95_ms,
        )
    speedups = np.asarray(
        [item.speedup_gdc_over_rgc for item in results], dtype=np.float64
    )
    rgc_faster = np.mean(
        np.asarray([item.rgc_minus_gdc_ms for item in results]) < 0
    )
    LOGGER.info(
        "paired result: median GDC/RGC speed ratio %.3fx; RGC faster on %.1f%% of frames",
        float(np.median(speedups)),
        100.0 * float(rgc_faster),
    )


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    validate_args(args, paths)
    configure_thread_environment(args.cpu_threads)
    algorithms = import_algorithms(args.repo_root)
    projection = load_projection_metadata(paths["projection_meta"])

    split_ids = read_frame_ids(paths["split_file"])
    available_ids = available_frame_ids(split_ids, paths)
    measured_ids = uniform_sample(available_ids, args.frames)
    warmup_ids = uniform_sample(available_ids, args.warmup_frames)
    LOGGER.info(
        "available=%d, measured=%d, warmup=%d, repeats=%d, CPU threads=%d",
        len(available_ids),
        len(measured_ids),
        len(warmup_ids),
        args.repeats,
        args.cpu_threads,
    )
    LOGGER.info(
        "scope: correction functions only; input loading and output saving are excluded"
    )
    LOGGER.info(
        "RGC configuration: residual_domain=%s, delta_clip=%s",
        args.residual_domain,
        args.delta_clip,
    )

    run_warmup(args, warmup_ids, paths, projection, algorithms)
    records = run_benchmark(args, measured_ids, paths, projection, algorithms)
    results = frame_results(records)
    summaries = summarize_methods(
        results, args.repeats, args.bootstrap_samples, args.seed + 1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataclass_csv(args.output_dir / "runtime_runs.csv", records)
    write_dataclass_csv(args.output_dir / "runtime_per_frame.csv", results)
    write_dataclass_csv(args.output_dir / "runtime_summary.csv", summaries)
    save_summary_json(
        args.output_dir / "runtime_summary.json", args, paths, summaries, results
    )
    save_runtime_plot(args.output_dir / "runtime_comparison", results)
    print_summary(summaries, results)
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        return run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
