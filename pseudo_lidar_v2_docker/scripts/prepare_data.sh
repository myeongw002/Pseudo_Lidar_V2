#!/usr/bin/env bash
set -euo pipefail
SCENEFLOW=${SCENEFLOW:-/data/SceneFlow}
KITTI=${KITTI:-/data/KITTI}

# README has a typo in the command name; the actual script is sceneflow.py.
python3 sceneflow.py --path "$SCENEFLOW" --force

# Convert KITTI velodyne points to depth maps for SDNet supervision.
python3 ./src/preprocess/generate_depth_map.py \
  --data_path "$KITTI/" \
  --split_file ./split/trainval.txt
