#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import sys
import torch
import numpy as np
print('python:', sys.version)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('torch cuda:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'gpu {i}:', torch.cuda.get_device_name(i), 'capability=', torch.cuda.get_device_capability(i))
print('numpy:', np.__version__)
PY
