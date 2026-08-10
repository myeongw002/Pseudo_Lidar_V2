#!/usr/bin/env bash
set -euo pipefail
KITTI=${KITTI:-/data/KITTI}
PRETRAIN=${PRETRAIN:-./results/sdn_sceneflow/checkpoint.pth.tar}
SAVE_PATH=${SAVE_PATH:-./results/sdn_kitti_trainval_set}
BTRAIN=${BTRAIN:-2}
BVAL=${BVAL:-1}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-300}

python3 ./src/main.py -c src/configs/sdn_kitti_train.config \
  --datapath "$KITTI/training/" \
  --pretrain "$PRETRAIN" \
  --save_path "$SAVE_PATH" \
  --split_train ./split/trainval.txt \
  --split_val ./split/subval.txt \
  --btrain "$BTRAIN" \
  --bval "$BVAL" \
  --workers "$WORKERS" \
  --epochs "$EPOCHS"
