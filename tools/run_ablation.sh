#!/usr/bin/env bash
set -euo pipefail

PYTHON=/usr/bin/python3
SCRIPT=/workspace/Pseudo_Lidar_V2/range_gdc/range_main_batch.py

ROOT=/data/kitti/pseudo_lidar_train
PRED_PATH=${ROOT}/range/raw_sdn/G64_range
ANCHOR_PATH=${ROOT}/anchor/range/G64_range
PROJECTION_META=${ROOT}/range/gt/meta/projection_meta.npz
ABLATION_ROOT=${ROOT}/range/ablation

run_ablation() {
    MODE="$1"
    OUT_ROOT="${ABLATION_ROOT}/${MODE}"

    mkdir -p \
        "${OUT_ROOT}/G64_range" \
        "${OUT_ROOT}/G64_mask" \
        "${OUT_ROOT}/meta"

    echo "========================================"
    echo "Running ablation mode: ${MODE}"
    echo "Output: ${OUT_ROOT}"
    echo "========================================"

    "${PYTHON}" "${SCRIPT}" \
        --pred_path "${PRED_PATH}" \
        --anchor_path "${ANCHOR_PATH}" \
        --output_path "${OUT_ROOT}/G64_range" \
        --mask_output_path "${OUT_ROOT}/G64_mask" \
        --projection_meta_path "${PROJECTION_META}" \
        --meta_dir "${OUT_ROOT}/meta" \
        --stats_csv "${OUT_ROOT}/meta/range_gdc_stats.csv" \
        --threads 4 \
        --overwrite \
        --ablation_mode "${MODE}" \
        --anchor_reject abs \
        --abs_error_thr 2.0 \
        --log_ratio_thr 0.4 \
        --anchor_force_policy accepted_only \
        --method cg \
        --range_min 0.1 \
        --range_max 80.0 \
        --lambda_anchor 300.0 \
        --lambda_prior 0.1 \
        --lambda_smooth 1.0 \
        --neighbor angular_grid8 \
        --edge_spatial_mode angular \
        --sigma_angular 0.01 \
        --sigma_log_range 0.3 \
        --transfer_k 1 \
        --transfer_neighbor_mode rowcol \
        --direct_weight_mode nearest \
        --confidence_mode nearest \
        --sigma_conf_pixel 2.0 \
        --sigma_conf_log_range 0.05 \
        --confidence_power 2.0 \
        --selection_mode confidence_hard \
        --confidence_high_thr 0.8 \
        --confidence_low_thr 0.2 \
        --delta_clip 0.3 \
        2>&1 | tee "${OUT_ROOT}/meta/run.log"
}

run_ablation graph_only
run_ablation direct_only
run_ablation full
