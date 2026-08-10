#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME=${IMAGE_NAME:-pseudo-lidar-v2:cu128}
DOCKERFILE=${DOCKERFILE:-Dockerfile}
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
