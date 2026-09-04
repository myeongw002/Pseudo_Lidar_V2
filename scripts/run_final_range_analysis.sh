#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Final Range Analysis
#
# mode:
#   main     : Final R4 analyses
#   density  : R2/R4/R8 range metrics
#   all      : both
#
# Usage:
#   ./scripts/run_final_range_analysis.sh main
#   ./scripts/run_final_range_analysis.sh density
#   ./scripts/run_final_range_analysis.sh all
# ============================================================

MODE="${1:-all}"

REPO=/workspace/Pseudo_Lidar_V2
KITTI=/data/kitti/kitti_object/training

FINAL=/data/kitti/pseudo_lidar_final_trainval
DENS=/data/kitti/pseudo_lidar_density_val_final

VAL="$REPO/split/val.txt"
THREADS=4
EXPECTED=3769

# ------------------------------------------------------------
# Existing final R4 artifacts
# ------------------------------------------------------------

RAW_DEPTH="$FINAL/sdn/depth_maps/trainval_final"

GT_RANGE="$FINAL/range/gt/G64_range"
META="$FINAL/range/gt/meta/projection_meta.npz"

RAW_RANGE="$FINAL/range/raw_sdn/G64_range"

ANCHOR_R4="$FINAL/anchor/range_shared_canonical/G64_range"

GDC_R4="$FINAL/range/original_gdc_naive/G64_range"
RGC_R4="$FINAL/range/range_gdc/G64_range"

# ------------------------------------------------------------
# Analysis outputs
# ------------------------------------------------------------

ANALYSIS="$FINAL/analysis/val"

OVERALL="$ANALYSIS/overall"
COMMON="$ANALYSIS/common_support"
BOUNDARY="$ANALYSIS/boundary_topology"

# ============================================================
# Utility
# ============================================================

require_dir()
{
    if [ ! -d "$1" ]; then
        echo "ERROR: missing directory:"
        echo "  $1"
        exit 1
    fi
}

require_file()
{
    if [ ! -f "$1" ]; then
        echo "ERROR: missing file:"
        echo "  $1"
        exit 1
    fi
}

check_inputs()
{
    require_file "$VAL"
    require_file "$META"

    require_dir "$KITTI/velodyne"
    require_dir "$KITTI/calib"
    require_dir "$KITTI/image_2"

    require_dir "$RAW_DEPTH"
    require_dir "$GT_RANGE"
    require_dir "$RAW_RANGE"
    require_dir "$ANCHOR_R4"
    require_dir "$GDC_R4"
    require_dir "$RGC_R4"

    N=$(grep -cve '^[[:space:]]*$' "$VAL")

    if [ "$N" -ne "$EXPECTED" ]; then
        echo "ERROR: val split has $N frames, expected $EXPECTED"
        exit 1
    fi
}

# ============================================================
# 1. Final R4 analyses
# ============================================================

run_main()
{
    echo
    echo "============================================================"
    echo "FINAL R4 RANGE ANALYSES"
    echo "============================================================"

    mkdir -p "$OVERALL" "$COMMON" "$BOUNDARY"

    # --------------------------------------------------------
    # A. Overall range metrics
    # --------------------------------------------------------

    echo
    echo "[1/3] Overall range metrics"

    python3 -B range_gdc/evaluate_range_metrics.py \
        --split_file "$VAL" \
        --gt_range_path "$GT_RANGE" \
        --guide_range_path "$ANCHOR_R4" \
        --source_range_path "$RAW_RANGE" \
        --method raw_sdn="$RAW_RANGE" \
        --method original_gdc="$GDC_R4" \
        --method range_gdc="$RGC_R4" \
        --raw_method raw_sdn \
        --metrics_csv "$OVERALL/range_metrics.csv" \
        --summary_csv "$OVERALL/range_summary.csv" \
        --eval_domain final_r4_val \
        --range_min 0.1 \
        --range_max 80.0 \
        --source_row_indices 5 7 9 11 \
        --source_rows 4 \
        --projection_meta_path "$META" \
        --expected_height 64 \
        --expected_width 1024 \
        --expected_frame_count "$EXPECTED" \
        --enable_leakage_check \
        --anchor_range_path "$ANCHOR_R4" \
        --leakage_method range_gdc \
        --leakage_output_csv "$OVERALL/leakage.csv" \
        --leakage_summary_csv "$OVERALL/leakage_summary.csv"

    # --------------------------------------------------------
    # B. Common pre-correction GDC support
    # --------------------------------------------------------

    echo
    echo "[2/3] Common GDC support evaluation"

    python3 -B tools/evaluate_gdc_common_support.py \
        --raw-depth-dir "$RAW_DEPTH" \
        --calib-dir "$KITTI/calib" \
        --raw-range-dir "$RAW_RANGE" \
        --gt-range-dir "$GT_RANGE" \
        --rgc-range-dir "$RGC_R4" \
        --gdc-range-dir "$GDC_R4" \
        --gdc-label original_gdc \
        --projection-meta-path "$META" \
        --split-file "$VAL" \
        --output-dir "$COMMON" \
        --source-rows 5 7 9 11 \
        --range-min 0.1 \
        --range-max 80.0 \
        --consider-range -0.1 3.0 \
        --mapping-tol 1e-5 \
        --boundary-log-thr 0.2 \
        --boundary-radius 1

    # --------------------------------------------------------
    # C. Boundary / Interior
    #    Boundary-distance
    #    Correction activity
    # --------------------------------------------------------

    echo
    echo "[3/3] Boundary + correction activity analysis"

    python3 -B tools/analyze_boundary_topology.py \
        --raw-range-dir "$RAW_RANGE" \
        --gt-range-dir "$GT_RANGE" \
        --rgc-range-dir "$RGC_R4" \
        --gdc-range-dir "$GDC_R4" \
        --gdc-label original_gdc \
        --anchor-range-dir "$ANCHOR_R4" \
        --projection-meta-path "$META" \
        --split-file "$VAL" \
        --output-dir "$BOUNDARY" \
        --source-rows 5 7 9 11 \
        --range-min 0.1 \
        --range-max 80.0 \
        --boundary-log-thr 0.2 \
        --boundary-radius 1 \
        --correction-tol 1e-3 \
        --neighbor angular_grid8 \
        --edge-spatial-mode angular \
        --sigma-angular 0.01 \
        --sigma-tangent 1.0 \
        --sigma-log-range 0.3

    echo
    echo "Final R4 analyses DONE."
}

# ============================================================
# Density helpers
#
# R2 = [5, 9]
# R4 = [5, 7, 9, 11]
# R8 = [4, 5, 6, 7, 8, 9, 10, 11]
#
# R4 reuses the existing final trainval artifacts.
# Only R2 / R8 need new correction artifacts.
# ============================================================

run_density_method()
{
    D="$1"

    if [ "$D" = "2" ]; then
        ROWS=(5 9)
    elif [ "$D" = "8" ]; then
        ROWS=(4 5 6 7 8 9 10 11)
    else
        echo "run_density_method supports only R2 / R8"
        exit 1
    fi

    ROOT="$DENS/r${D}"

    SHARED_PC="$ROOT/anchor/shared_canonical_pointcloud"
    SHARED_INDEX="$ROOT/anchor/shared_canonical_source_index"
    SHARED_PROV="$ROOT/anchor/shared_canonical_pointcloud_provenance.json"

    IMAGE_ANCHOR="$ROOT/anchor/shared_canonical_image_depth"
    IMAGE_INDEX="$ROOT/anchor/shared_canonical_image_source_index"

    RANGE_ANCHOR="$ROOT/anchor/range_shared_canonical/G64_range"
    RANGE_MASK="$ROOT/anchor/range_shared_canonical/G64_mask"
    RANGE_META="$ROOT/anchor/range_shared_canonical/meta"

    GDC_DEPTH="$ROOT/gdc/corrected_depth"
    GDC_RANGE_ROOT="$ROOT/gdc/range"

    RGC_RANGE="$ROOT/rgc/G64_range"
    RGC_MASK="$ROOT/rgc/G64_mask"
    RGC_META="$ROOT/rgc/meta"

    DENS_META="$ROOT/projection_meta.npz"

    mkdir -p \
        "$SHARED_PC" \
        "$SHARED_INDEX" \
        "$IMAGE_ANCHOR" \
        "$IMAGE_INDEX" \
        "$RANGE_ANCHOR" \
        "$RANGE_MASK" \
        "$RANGE_META" \
        "$GDC_DEPTH" \
        "$GDC_RANGE_ROOT" \
        "$RGC_RANGE" \
        "$RGC_MASK" \
        "$RGC_META"

    echo
    echo "============================================================"
    echo "R${D}: rows ${ROWS[*]}"
    echo "============================================================"

    # --------------------------------------------------------
    # 1. Exact physical sparse point set
    # --------------------------------------------------------

    echo "[R${D} 1/6] shared sparse LiDAR"

    python3 -B tools/create_shared_canonical_anchor.py \
        --velodyne-dir "$KITTI/velodyne" \
        --split-file "$VAL" \
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
    # 2. GDC camera-depth anchor
    # --------------------------------------------------------

    echo "[R${D} 2/6] sparse point -> image depth"

    python3 -B gdc/ptc2depthmap.py \
        --input_path "$SHARED_PC" \
        --output_path "$IMAGE_ANCHOR" \
        --calib_path "$KITTI/calib" \
        --image_path "$KITTI/image_2" \
        --split_file "$VAL" \
        --threads "$THREADS" \
        --collision-policy nearest_positive \
        --provenance-json "$SHARED_PROV" \
        --source-index-input-path "$SHARED_INDEX" \
        --source-index-output-path "$IMAGE_INDEX" \
        --selected-rows "${ROWS[@]}"

    # --------------------------------------------------------
    # 3. RGC range anchor
    # --------------------------------------------------------

    echo "[R${D} 3/6] sparse point -> range anchor"

    python3 -B range_gdc/build_range_anchor.py \
        --output_range_path "$RANGE_ANCHOR" \
        --output_mask_path "$RANGE_MASK" \
        --split_file "$VAL" \
        --selected_rows "${ROWS[@]}" \
        --invalid_value 0.0 \
        --expected_height 64 \
        --expected_width 1024 \
        --projection_meta_path "$META" \
        --meta_dir "$RANGE_META" \
        --threads "$THREADS" \
        --source-index-dir "$SHARED_INDEX" \
        --shared-pointcloud-dir "$SHARED_PC" \
        --source-provenance-path "$SHARED_PROV"

    # --------------------------------------------------------
    # 4. Final RGC
    # --------------------------------------------------------

    echo "[R${D} 4/6] RGC"

    python3 -B range_gdc/range_main_batch.py \
        --pred_path "$RAW_RANGE" \
        --anchor_path "$RANGE_ANCHOR" \
        --split_file "$VAL" \
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

    # --------------------------------------------------------
    # 5. Original GDC
    # --------------------------------------------------------

    echo "[R${D} 5/6] Original GDC"

    python3 -B gdc/main_batch.py \
        --input_path "$RAW_DEPTH" \
        --calib_path "$KITTI/calib" \
        --gt_depthmap_path "$IMAGE_ANCHOR" \
        --output_path "$GDC_DEPTH" \
        --split_file "$VAL" \
        --threads "$THREADS" \
        --stats_csv "$ROOT/gdc/gdc_stats.csv" \
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
    # 6. GDC depth -> canonical range grid
    #
    # IMPORTANT:
    # use density-specific projection_meta.
    # Do NOT overwrite the final R4 META.
    # --------------------------------------------------------

    echo "[R${D} 6/6] GDC depth -> range"

    python3 -B range_gdc/range_projection.py depth-to-range \
        --input_path "$GDC_DEPTH" \
        --output_path "$GDC_RANGE_ROOT" \
        --calib_path "$KITTI/calib" \
        --split_file "$VAL" \
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
        --meta_path "$DENS_META" \
        --stats_csv "$GDC_RANGE_ROOT/meta/projection_stats.csv"

    echo "[R${D}] DONE"
}

run_density()
{
    echo
    echo "============================================================"
    echo "R2 / R4 / R8 RANGE DENSITY EXPERIMENT"
    echo "============================================================"

    mkdir -p "$DENS/eval"

    # R4 already exists as the final trainval result.
    # Generate only missing densities.
    run_density_method 2
    run_density_method 8

    GDC_R2="$DENS/r2/gdc/range/G64_range"
    RGC_R2="$DENS/r2/rgc/G64_range"

    GDC_R8="$DENS/r8/gdc/range/G64_range"
    RGC_R8="$DENS/r8/rgc/G64_range"

    # --------------------------------------------------------
    # IMPORTANT FAIRNESS:
    #
    # Every density is evaluated after excluding the UNION
    # of all R8 source rows:
    #
    #     [4,5,6,7,8,9,10,11]
    #
    # Do NOT pass the final R4 projection_meta here.
    # evaluate_range_metrics gives metadata selected_rows
    # precedence over explicit --source_row_indices.
    # --------------------------------------------------------

    echo
    echo "[DENSITY] Common-domain range evaluation"

    python3 -B range_gdc/evaluate_range_metrics.py \
        --split_file "$VAL" \
        --gt_range_path "$GT_RANGE" \
        --guide_range_path "$ANCHOR_R4" \
        --source_range_path "$RAW_RANGE" \
        --method raw_sdn="$RAW_RANGE" \
        --method gdc_r2="$GDC_R2" \
        --method rgc_r2="$RGC_R2" \
        --method gdc_r4="$GDC_R4" \
        --method rgc_r4="$RGC_R4" \
        --method gdc_r8="$GDC_R8" \
        --method rgc_r8="$RGC_R8" \
        --raw_method raw_sdn \
        --metrics_csv "$DENS/eval/range_metrics.csv" \
        --summary_csv "$DENS/eval/range_summary.csv" \
        --eval_domain density_r2_r4_r8_common \
        --range_min 0.1 \
        --range_max 80.0 \
        --source_row_indices 4 5 6 7 8 9 10 11 \
        --source_rows 8 \
        --expected_height 64 \
        --expected_width 1024 \
        --expected_frame_count "$EXPECTED"

    echo
    echo "Density range evaluation DONE."
}

# ============================================================
# Result print
# ============================================================

print_results()
{
    echo
    echo "============================================================"
    echo "RESULT SUMMARY"
    echo "============================================================"

    if [ -f "$OVERALL/range_summary.csv" ]; then
        echo
        echo "--- Final R4 overall / common_hidden_valid ---"

        python3 - <<'PY'
import csv

path = "/data/kitti/pseudo_lidar_final_trainval/analysis/val/overall/range_summary.csv"

with open(path) as f:
    rows = list(csv.DictReader(f))

order = ["raw_sdn", "original_gdc", "range_gdc"]

selected = {
    r["method"]: r
    for r in rows
    if r["area"] == "common_hidden_valid"
}

print(
    f"{'method':16s} "
    f"{'MAE':>10s} {'RMSE':>10s} "
    f"{'Median':>10s} {'P90':>10s} {'P95':>10s}"
)

for method in order:
    r = selected[method]
    print(
        f"{method:16s} "
        f"{float(r['mae_weighted']):10.6f} "
        f"{float(r['rmse_weighted']):10.6f} "
        f"{float(r['median_abs_mean']):10.6f} "
        f"{float(r['p90_abs_mean']):10.6f} "
        f"{float(r['p95_abs_mean']):10.6f}"
    )
PY
    fi

    if [ -f "$DENS/eval/range_summary.csv" ]; then
        echo
        echo "--- Density R2/R4/R8 / common_hidden_valid ---"

        python3 - <<'PY'
import csv

path = "/data/kitti/pseudo_lidar_density_val_final/eval/range_summary.csv"

with open(path) as f:
    rows = list(csv.DictReader(f))

order = [
    "raw_sdn",
    "gdc_r2", "rgc_r2",
    "gdc_r4", "rgc_r4",
    "gdc_r8", "rgc_r8",
]

selected = {
    r["method"]: r
    for r in rows
    if r["area"] == "common_hidden_valid"
}

print(
    f"{'method':12s} "
    f"{'MAE':>10s} {'RMSE':>10s} "
    f"{'Median':>10s} {'P90':>10s} {'P95':>10s}"
)

for method in order:
    r = selected[method]
    print(
        f"{method:12s} "
        f"{float(r['mae_weighted']):10.6f} "
        f"{float(r['rmse_weighted']):10.6f} "
        f"{float(r['median_abs_mean']):10.6f} "
        f"{float(r['p90_abs_mean']):10.6f} "
        f"{float(r['p95_abs_mean']):10.6f}"
    )
PY
    fi

    echo
    echo "Analysis outputs:"
    echo "  Overall         : $OVERALL"
    echo "  Common support  : $COMMON"
    echo "  Boundary        : $BOUNDARY"
    echo "  Density         : $DENS/eval"
}

# ============================================================
# Main
# ============================================================

check_inputs

case "$MODE" in

    main)
        run_main
        ;;

    density)
        run_density
        ;;

    all)
        run_main
        run_density
        ;;

    *)
        echo "Usage:"
        echo "  $0 main"
        echo "  $0 density"
        echo "  $0 all"
        exit 1
        ;;

esac

print_results

echo
echo "DONE."
