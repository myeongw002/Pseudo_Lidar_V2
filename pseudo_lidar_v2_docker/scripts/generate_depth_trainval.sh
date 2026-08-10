#!/usr/bin/env bash
set -euo pipefail
KITTI=${KITTI:-/data/KITTI}
RESUME=${RESUME:-./results/sdn_kitti_train_set/checkpoint.pth.tar}
SAVE_PATH=${SAVE_PATH:-./results/sdn_kitti_train_set}
BVAL=${BVAL:-1}
WORKERS=${WORKERS:-4}

python3 ./src/main.py -c src/configs/sdn_kitti_train.config \
  --resume "$RESUME" \
  --datapath "$KITTI/training/" \
  --save_path "$SAVE_PATH" \
  --data_list ./split/trainval.txt \
  --generate_depth_map \
  --data_tag trainval \
  --bval "$BVAL" \
  --workers "$WORKERS"
