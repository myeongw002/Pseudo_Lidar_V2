#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Final trainval preprocessing
#
# Methods
#   1. Raw SDN
#   2. Original GDC (naive)
#   3. Final RGC
#
# Final RGC
#   sparse rows      : [5, 7, 9, 11]
#   residual         : linear
#   lambda_anchor    : 100
#   lambda_prior     : 0.1
#   lambda_smooth    : 1.0
#   topology         : angular_grid8
#   range gate       : log_gaussian
#   delta clipping   : disabled
#
# Usage
#   ./scripts/run_trainval_final.sh all
#   ./scripts/run_trainval_final.sh raw
#   ./scripts/run_trainval_final.sh gdc
#   ./scripts/run_trainval_final.sh rgc
#   ./scripts/run_trainval_final.sh pointcloud
#
# ============================================================

MODE="${1:-all}"

REPO=/workspace/Pseudo_Lidar_V2

KITTI=/data/kitti/kitti_object/training
SPLIT="$REPO/split/trainval.txt"

OUT=/data/kitti/pseudo_lidar_final_trainval

THREADS=4
ROWS=(5 7 9 11)

SDN_CONFIG="$REPO/src/configs/sdn_kitti_train.config"
SDN_CKPT=/data/kitti/pseudo_lidar/sdn_kitti_train_set/model_best.pth.tar
DATA_TAG=trainval_final

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SDN_DEPTH="$OUT/sdn/depth_maps/$DATA_TAG"

SHARED_PC="$OUT/anchor/shared_canonical_pointcloud"
SHARED_INDEX="$OUT/anchor/shared_canonical_source_index"
SHARED_PROV="$OUT/anchor/shared_canonical_pointcloud_provenance.json"

ANCHOR_IMAGE_DEPTH="$OUT/anchor/shared_canonical_image_depth"
ANCHOR_IMAGE_INDEX="$OUT/anchor/shared_canonical_image_source_index"

ANCHOR_RANGE_ROOT="$OUT/anchor/range_shared_canonical"
ANCHOR_RANGE="$ANCHOR_RANGE_ROOT/G64_range"
ANCHOR_MASK="$ANCHOR_RANGE_ROOT/G64_mask"
ANCHOR_META="$ANCHOR_RANGE_ROOT/meta"

GT_ROOT="$OUT/range/gt"
GT_RANGE="$GT_ROOT/G64_range"
META="$GT_ROOT/meta/projection_meta.npz"

RAW_ROOT="$OUT/range/raw_sdn"
RAW_RANGE="$RAW_ROOT/G64_range"

GDC_DEPTH="$OUT/original_gdc/naive/corrected_depth"
GDC_STATS="$OUT/original_gdc/naive/stats/gdc_stats.csv"

GDC_RANGE_ROOT="$OUT/range/original_gdc_naive"
GDC_RANGE="$GDC_RANGE_ROOT/G64_range"

RGC_ROOT="$OUT/range/range_gdc"
RGC_RANGE="$RGC_ROOT/G64_range"
RGC_MASK="$RGC_ROOT/G64_mask"
RGC_META="$RGC_ROOT/meta"

PC_RAW="$OUT/pointcloud/raw_sdn"
PC_GDC="$OUT/pointcloud/original_gdc"
PC_RGC="$OUT/pointcloud/rgc"

# ============================================================
# Utility
# ============================================================

count_split()
{
    grep -cve '^[[:space:]]*$' "$SPLIT"
}

count_npy()
{
    local DIR="$1"

    if [ ! -d "$DIR" ]; then
        echo 0
        return
    fi

    find "$DIR" -maxdepth 1 -name '*.npy' | wc -l
}

count_bin()
{
    local DIR="$1"

    if [ ! -d "$DIR" ]; then
        echo 0
        return
    fi

    find "$DIR" -maxdepth 1 -name '*.bin' | wc -l
}

require_dir()
{
    local DIR="$1"

    if [ ! -d "$DIR" ]; then
        echo "ERROR: missing directory:"
        echo "  $DIR"
        exit 1
    fi
}

require_file()
{
    local FILE="$1"

    if [ ! -f "$FILE" ]; then
        echo "ERROR: missing file:"
        echo "  $FILE"
        exit 1
    fi
}

# ============================================================
# Input validation
# ============================================================

cd "$REPO"

require_file "$SPLIT"
require_file "$SDN_CONFIG"
require_file "$SDN_CKPT"

require_dir "$KITTI/velodyne"
require_dir "$KITTI/calib"
require_dir "$KITTI/image_2"

EXPECTED=$(count_split)

echo "============================================================"
echo "Final trainval preprocessing"
echo "============================================================"
echo "mode       : $MODE"
echo "output     : $OUT"
echo "split      : $SPLIT"
echo "frames     : $EXPECTED"
echo "rows       : ${ROWS[*]}"
echo "============================================================"

if [ "$EXPECTED" -ne 7481 ]; then
    echo "WARNING: expected KITTI trainval=7481 but split has $EXPECTED"
fi

mkdir -p "$OUT"

# ============================================================
# Shared preprocessing
# ============================================================

run_shared()
{
    echo
    echo "============================================================"
    echo "[SHARED] Sparse canonical LiDAR anchors"
    echo "============================================================"

    mkdir -p \
        "$SHARED_PC" \
        "$SHARED_INDEX" \
        "$ANCHOR_IMAGE_DEPTH" \
        "$ANCHOR_IMAGE_INDEX" \
        "$ANCHOR_RANGE" \
        "$ANCHOR_MASK" \
        "$ANCHOR_META" \
        "$GT_ROOT"

    # --------------------------------------------------------
    # 1. Canonical sparse physical LiDAR points
    # --------------------------------------------------------

    echo "[SHARED 1/4] canonical sparse point cloud"

    python3 -B tools/create_shared_canonical_anchor.py \
        --velodyne-dir "$KITTI/velodyne" \
        --split-file "$SPLIT" \
        --output-pointcloud-dir "$SHARED_PC" \
        --output-source-index-dir "$SHARED_INDEX" \
        --provenance-json "$SHARED_PROV" \
        --height 64 \
        --width 1024 \
        --vmin-deg -24.9 \
        --vmax-deg 2.0 \
        --azimuth-mode full_360_front_centered \
        --range-min 0.1 \
        --range-max 80.0 \
        --invalid-value 0.0 \
        --selected-rows "${ROWS[@]}"

    # --------------------------------------------------------
    # 2. Projection metadata + GT range
    #
    # GT is generated now because it will also be reused later
    # for range metrics / correction activity / boundary analysis.
    # --------------------------------------------------------

    echo "[SHARED 2/4] GT range + projection metadata"

    python3 -B range_gdc/range_projection.py ptc-to-range \
        --input_path "$KITTI/velodyne" \
        --output_path "$GT_ROOT" \
        --calib_path "$KITTI/calib" \
        --image_path "$KITTI/image_2" \
        --split_file "$SPLIT" \
        --threads "$THREADS" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --azimuth_mode full_360_front_centered \
        --depth_min 0.1 \
        --depth_max 80.0 \
        --invalid_value 0.0 \
        --anchor_rows "${ROWS[@]}" \
        --meta_path "$META" \
        --stats_csv "$GT_ROOT/meta/projection_stats.csv" \
        --image_fov_only

    # --------------------------------------------------------
    # 3. Same physical sparse points -> camera-depth GDC input
    # --------------------------------------------------------

    echo "[SHARED 3/4] sparse anchor -> camera depth"

    python3 -B gdc/ptc2depthmap.py \
        --input_path "$SHARED_PC" \
        --output_path "$ANCHOR_IMAGE_DEPTH" \
        --calib_path "$KITTI/calib" \
        --image_path "$KITTI/image_2" \
        --split_file "$SPLIT" \
        --threads "$THREADS" \
        --collision-policy nearest_positive \
        --provenance-json "$SHARED_PROV" \
        --source-index-input-path "$SHARED_INDEX" \
        --source-index-output-path "$ANCHOR_IMAGE_INDEX" \
        --selected-rows "${ROWS[@]}"

    # --------------------------------------------------------
    # 4. Same physical sparse points -> RGC range anchor
    # --------------------------------------------------------

    echo "[SHARED 4/4] sparse anchor -> range grid"

    python3 -B range_gdc/build_range_anchor.py \
        --output_range_path "$ANCHOR_RANGE" \
        --output_mask_path "$ANCHOR_MASK" \
        --split_file "$SPLIT" \
        --selected_rows "${ROWS[@]}" \
        --invalid_value 0.0 \
        --expected_height 64 \
        --expected_width 1024 \
        --projection_meta_path "$META" \
        --meta_dir "$ANCHOR_META" \
        --threads "$THREADS" \
        --source-index-dir "$SHARED_INDEX" \
        --shared-pointcloud-dir "$SHARED_PC" \
        --source-provenance-path "$SHARED_PROV"

    echo "[SHARED] done"
}

# ============================================================
# Raw SDN
# ============================================================

run_raw()
{
    echo
    echo "============================================================"
    echo "[RAW] SDN -> range"
    echo "============================================================"

    mkdir -p "$OUT/sdn" "$RAW_ROOT"

    # --------------------------------------------------------
    # 1. SDN stereo depth
    # --------------------------------------------------------

    echo "[RAW 1/2] generate SDN depth"

    python3 -B src/main.py \
        -c "$SDN_CONFIG" \
        --resume "$SDN_CKPT" \
        --datapath "$KITTI" \
        --data_list "$SPLIT" \
        --generate_depth_map \
        --data_tag "$DATA_TAG" \
        --save_path "$OUT/sdn" \
        --workers "$THREADS" \
        --bval 1

    require_dir "$SDN_DEPTH"

    # --------------------------------------------------------
    # 2. SDN depth -> canonical range grid
    # --------------------------------------------------------

    echo "[RAW 2/2] SDN depth -> range grid"

    python3 -B range_gdc/range_projection.py depth-to-range \
        --input_path "$SDN_DEPTH" \
        --output_path "$RAW_ROOT" \
        --calib_path "$KITTI/calib" \
        --split_file "$SPLIT" \
        --threads "$THREADS" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --azimuth_mode full_360_front_centered \
        --depth_min 0.1 \
        --depth_max 80.0 \
        --invalid_value 0.0 \
        --anchor_rows "${ROWS[@]}" \
        --meta_path "$META" \
        --stats_csv "$RAW_ROOT/meta/projection_stats.csv"

    echo "[RAW] done"
}

# ============================================================
# Original GDC
# ============================================================

run_gdc()
{
    echo
    echo "============================================================"
    echo "[GDC] Original GDC naive"
    echo "============================================================"

    require_dir "$SDN_DEPTH"
    require_dir "$ANCHOR_IMAGE_DEPTH"
    require_file "$META"

    mkdir -p \
        "$GDC_DEPTH" \
        "$(dirname "$GDC_STATS")" \
        "$GDC_RANGE_ROOT"

    # --------------------------------------------------------
    # 1. Original GDC
    # --------------------------------------------------------

    echo "[GDC 1/2] camera-depth graph correction"

    python3 -B gdc/main_batch.py \
        --input_path "$SDN_DEPTH" \
        --calib_path "$KITTI/calib" \
        --gt_depthmap_path "$ANCHOR_IMAGE_DEPTH" \
        --output_path "$GDC_DEPTH" \
        --split_file "$SPLIT" \
        --threads "$THREADS" \
        --stats_csv "$GDC_STATS" \
        --k 10 \
        --recon_tol 0.0005 \
        --method cg \
        --consider_range -0.1 3.0 \
        --disable_subsample \
        --anchor_reject abs \
        --abs_error_thr 2.0 \
        --log_ratio_thr 0.4 \
        --anchor_force_policy accepted_only

    # --------------------------------------------------------
    # 2. corrected camera depth -> canonical range grid
    # --------------------------------------------------------

    echo "[GDC 2/2] corrected depth -> range grid"

    python3 -B range_gdc/range_projection.py depth-to-range \
        --input_path "$GDC_DEPTH" \
        --output_path "$GDC_RANGE_ROOT" \
        --calib_path "$KITTI/calib" \
        --split_file "$SPLIT" \
        --threads "$THREADS" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --azimuth_mode full_360_front_centered \
        --depth_min 0.1 \
        --depth_max 80.0 \
        --invalid_value 0.0 \
        --anchor_rows "${ROWS[@]}" \
        --meta_path "$META" \
        --stats_csv "$GDC_RANGE_ROOT/meta/projection_stats.csv"

    echo "[GDC] done"
}

# ============================================================
# Final RGC
# ============================================================

run_rgc()
{
    echo
    echo "============================================================"
    echo "[RGC] Final linear Range-GDC"
    echo "============================================================"

    require_dir "$RAW_RANGE"
    require_dir "$ANCHOR_RANGE"
    require_file "$META"

    mkdir -p \
        "$RGC_RANGE" \
        "$RGC_MASK" \
        "$RGC_META"

    python3 -B range_gdc/range_main_batch.py \
        --pred_path "$RAW_RANGE" \
        --anchor_path "$ANCHOR_RANGE" \
        --split_file "$SPLIT" \
        --output_path "$RGC_RANGE" \
        --mask_output_path "$RGC_MASK" \
        --projection_meta_path "$META" \
        --meta_dir "$RGC_META" \
        --stats_csv "$RGC_META/range_gdc_stats.csv" \
        --method cg \
        --range_min 0.1 \
        --range_max 80.0 \
        --anchor_reject abs \
        --abs_error_thr 2.0 \
        --log_ratio_thr 0.4 \
        --lambda_anchor 100 \
        --lambda_prior 0.1 \
        --lambda_smooth 1.0 \
        --neighbor angular_grid8 \
        --edge_spatial_mode angular \
        --edge_range_mode log_gaussian \
        --sigma_angular 0.01 \
        --sigma_tangent 1.0 \
        --sigma_log_range 0.3 \
        --residual_domain linear \
        --disable_delta_clip \
        --anchor_force_policy accepted_only \
        --threads "$THREADS"

    echo "[RGC] done"
}

# ============================================================
# Export detector point clouds
# ============================================================

run_pointcloud()
{
    echo
    echo "============================================================"
    echo "[POINTCLOUD] Raw / GDC / RGC"
    echo "============================================================"

    require_file "$META"
    require_dir "$RAW_RANGE"
    require_dir "$GDC_RANGE"
    require_dir "$RGC_RANGE"

    mkdir -p "$PC_RAW" "$PC_GDC" "$PC_RGC"

    echo "[POINTCLOUD 1/3] Raw SDN"

    python3 -B range_gdc/range_to_pointcloud.py \
        --range_path "$RAW_RANGE" \
        --output_path "$PC_RAW" \
        --split_file "$SPLIT" \
        --projection_meta_path "$META" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --range_min 0.1 \
        --range_max 80.0

    echo "[POINTCLOUD 2/3] Original GDC"

    python3 -B range_gdc/range_to_pointcloud.py \
        --range_path "$GDC_RANGE" \
        --output_path "$PC_GDC" \
        --split_file "$SPLIT" \
        --projection_meta_path "$META" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --range_min 0.1 \
        --range_max 80.0

    echo "[POINTCLOUD 3/3] Final RGC"

    python3 -B range_gdc/range_to_pointcloud.py \
        --range_path "$RGC_RANGE" \
        --output_path "$PC_RGC" \
        --split_file "$SPLIT" \
        --projection_meta_path "$META" \
        --height 64 \
        --width 1024 \
        --vmin_deg -24.9 \
        --vmax_deg 2.0 \
        --range_min 0.1 \
        --range_max 80.0

    echo "[POINTCLOUD] done"
}

# ============================================================
# Summary
# ============================================================

summary()
{
    echo
    echo "============================================================"
    echo "OUTPUT SUMMARY"
    echo "============================================================"

    printf "%-30s %8s / %s\n" \
        "Shared sparse PCD" "$(count_bin "$SHARED_PC")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "SDN depth" "$(count_npy "$SDN_DEPTH")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "Raw range" "$(count_npy "$RAW_RANGE")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "GDC depth" "$(count_npy "$GDC_DEPTH")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "GDC range" "$(count_npy "$GDC_RANGE")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "RGC range" "$(count_npy "$RGC_RANGE")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "Raw pointcloud" "$(count_bin "$PC_RAW")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "GDC pointcloud" "$(count_bin "$PC_GDC")" "$EXPECTED"

    printf "%-30s %8s / %s\n" \
        "RGC pointcloud" "$(count_bin "$PC_RGC")" "$EXPECTED"

    echo "============================================================"
}

# ============================================================
# Mode
# ============================================================

case "$MODE" in

    all)
        run_shared
        run_raw
        run_gdc
        run_rgc
        run_pointcloud
        ;;

    shared)
        run_shared
        ;;

    raw)
        if [ ! -f "$META" ]; then
            echo "Projection metadata missing -> running shared first."
            run_shared
        fi
        run_raw
        ;;

    gdc)
        run_gdc
        ;;

    rgc)
        run_rgc
        ;;

    pointcloud)
        run_pointcloud
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo
        echo "Usage:"
        echo "  $0 all"
        echo "  $0 shared"
        echo "  $0 raw"
        echo "  $0 gdc"
        echo "  $0 rgc"
        echo "  $0 pointcloud"
        exit 1
        ;;

esac

summary

echo
echo "DONE."
