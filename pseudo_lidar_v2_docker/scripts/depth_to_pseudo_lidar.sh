#!/usr/bin/env bash
set -euo pipefail
KITTI=${KITTI:-/data/KITTI}
DEPTH_DIR=${DEPTH_DIR:-./results/sdn_kitti_train_set/depth_maps/trainval/}
SAVE_DIR=${SAVE_DIR:-./results/sdn_kitti_train_set/pseudo_lidar_trainval/}

python3 ./src/preprocess/generate_lidar_from_depth.py \
  --calib_dir "$KITTI/training/calib" \
  --depth_dir "$DEPTH_DIR" \
  --save_dir "$SAVE_DIR"
