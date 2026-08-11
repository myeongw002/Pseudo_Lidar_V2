#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME=${IMAGE_NAME:-pseudo-lidar-v2:cu128}
REPO_DIR=${REPO_DIR:-$PWD/..}
KITTI_DIR=${KITTI_DIR:-$PWD/data/KITTI}
SCENEFLOW_DIR=${SCENEFLOW_DIR:-$PWD/data/SceneFlow}
RESULTS_DIR=${RESULTS_DIR:-$PWD/results}
mkdir -p "$RESULTS_DIR"
docker run --gpus all --ipc=host --shm-size=16g --rm -it \
  -v "$REPO_DIR:/workspace/Pseudo_Lidar_V2" \
  -v "$KITTI_DIR:/data/kitti" \
  -v "$RESULTS_DIR:/workspace/Pseudo_Lidar_V2/results" \
  -e PYTHONPATH=/opt/plv2_compat:/workspace/Pseudo_Lidar_V2/src:/workspace/Pseudo_Lidar_V2 \
  -w /workspace/Pseudo_Lidar_V2 \
  "$IMAGE_NAME" bash
