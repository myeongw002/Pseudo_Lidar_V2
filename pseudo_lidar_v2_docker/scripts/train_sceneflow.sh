#!/usr/bin/env bash
set -euo pipefail
BTRAIN=${BTRAIN:-2}
BVAL=${BVAL:-1}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-10}

python3 ./src/main.py -c src/configs/sdn_sceneflow.config \
  --datapath ./sceneflow/ \
  --save_path ./results/sdn_sceneflow \
  --btrain "$BTRAIN" \
  --bval "$BVAL" \
  --workers "$WORKERS" \
  --epochs "$EPOCHS"
