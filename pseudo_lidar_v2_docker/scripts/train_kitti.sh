#!/usr/bin/env bash
set -euo pipefail
KITTI=${KITTI:-/data/KITTI}
PRETRAIN=${PRETRAIN:-./results/sdn_sceneflow/checkpoint.pth.tar}
SAVE_PATH=${SAVE_PATH:-./results/sdn_kitti_train_set}
SPLIT_TRAIN=${SPLIT_TRAIN:-./split/train.txt}
SPLIT_VAL=${SPLIT_VAL:-./split/subval.txt}
BTRAIN=${BTRAIN:-2}
BVAL=${BVAL:-1}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-300}

python3 ./src/main.py -c src/configs/sdn_kitti_train.config \
  --datapath "$KITTI/training/" \
  --pretrain "$PRETRAIN" \
  --save_path "$SAVE_PATH" \
  --split_train "$SPLIT_TRAIN" \
  --split_val "$SPLIT_VAL" \
  --btrain "$BTRAIN" \
  --bval "$BVAL" \
  --workers "$WORKERS" \
  --epochs "$EPOCHS"
