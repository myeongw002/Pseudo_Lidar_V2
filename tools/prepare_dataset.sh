#!/usr/bin/env bash
set -euo pipefail

KITTI_ROOT=/media/myungw00/2TB_SSD/kitti/kitti_object
PL_ROOT=/media/myungw00/2TB_SSD/kitti/pseudo_lidar_train
VIEW_ROOT=/media/myungw00/2TB_SSD/kitti/pointpillars_views
IMAGESETS=/home/myungw00/ROS2/upsample_ws/pointpillar_ws/pointpillars_openpcdet_docker/runtime/kitti/ImageSets

for d in \
    raw_sdn_64ch \
    original_gdc_naive_64ch \
    range_gdc_64ch
do
    path="${PL_ROOT}/pointcloud/${d}"
    printf "%-35s %s\n" "${d}" \
        "$(find "${path}" -maxdepth 1 -name '*.bin' | wc -l)"
done

make_view() {
    local name="$1"
    local velodyne_path="$2"
    local root="${VIEW_ROOT}/${name}"

    mkdir -p "${root}/training"

    ln -sfn "${IMAGESETS}"                  "${root}/ImageSets"
    ln -sfn "${velodyne_path}"              "${root}/training/velodyne"
    ln -sfn "${KITTI_ROOT}/training/calib"  "${root}/training/calib"
    ln -sfn "${KITTI_ROOT}/training/label_2" "${root}/training/label_2"
    ln -sfn "${KITTI_ROOT}/training/image_2" "${root}/training/image_2"

    if [[ -d "${KITTI_ROOT}/training/planes" ]]; then
        ln -sfn "${KITTI_ROOT}/training/planes" "${root}/training/planes"
    fi
}

make_view \
    raw_sdn \
    "${PL_ROOT}/pointcloud/raw_sdn_64ch"

make_view \
    gdc_naive \
    "${PL_ROOT}/pointcloud/original_gdc_naive_64ch"

make_view \
    rgc \
    "${PL_ROOT}/pointcloud/range_gdc_64ch"

make_view \
    lidar64 \
    "${KITTI_ROOT}/training/velodyne"

tree -L 3 "${VIEW_ROOT}"




