# Pseudo-LiDAR++ / Pseudo_Lidar_V2 training image for modern NVIDIA GPUs
# Put this Dockerfile in the Pseudo_Lidar_V2 repo root, or run with the repo mounted at /workspace/Pseudo_Lidar_V2.
# Recommended for RTX 40/50-class GPUs. For exact 2020 reproduction, see Dockerfile.legacy.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    WORKDIR=/workspace/Pseudo_Lidar_V2

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 12.8 wheel. Leave versions unpinned by default so the build follows
# the current stable wheel set published by PyTorch for cu128.
# If exact reproducibility is needed, pass for example:
#   --build-arg TORCH_SPEC='torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1'
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC="torch torchvision torchaudio"
RUN python3 -m pip install ${TORCH_SPEC} --index-url ${PYTORCH_INDEX_URL}

COPY requirements-modern.txt /tmp/requirements-modern.txt
RUN python3 -m pip install -r /tmp/requirements-modern.txt

# Compatibility shim for old code that may still reference removed numpy aliases.
RUN mkdir -p /opt/plv2_compat && cat > /opt/plv2_compat/sitecustomize.py <<'PY'
import numpy as _np
_aliases = {
    'bool': bool,
    'int': int,
    'float': float,
    'complex': complex,
    'object': object,
    'str': str,
}
for _name, _typ in _aliases.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _typ)
PY

ENV PYTHONPATH=/opt/plv2_compat:/workspace/Pseudo_Lidar_V2/src
WORKDIR /workspace/Pseudo_Lidar_V2

# Patch helper for modern PyTorch. Run once inside a mounted repo:
#   plv2_patch_repo
RUN cat > /usr/local/bin/plv2_patch_repo <<'SH' && chmod +x /usr/local/bin/plv2_patch_repo
#!/usr/bin/env bash
set -euo pipefail
cd "${1:-/workspace/Pseudo_Lidar_V2}"
python3 - <<'PY'
from pathlib import Path
p = Path('src/main.py')
if not p.exists():
    raise SystemExit('src/main.py not found. Mount or copy the Pseudo_Lidar_V2 repo at /workspace/Pseudo_Lidar_V2')
s = p.read_text()
# PyTorch >=2.6 changed torch.load default behavior. Old checkpoints with optimizer/scheduler state
# should be loaded with weights_only=False.
s = s.replace('torch.load(args.pretrain)', 'torch.load(args.pretrain, weights_only=False)')
s = s.replace('torch.load(args.resume)', 'torch.load(args.resume, weights_only=False)')
p.write_text(s)
print('Patched src/main.py for modern torch.load compatibility.')
PY
SH

CMD ["bash"]
