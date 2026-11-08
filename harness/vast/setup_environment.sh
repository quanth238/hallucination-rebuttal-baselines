#!/usr/bin/env bash
set -euo pipefail
source /workspace/rebuttal/control/vast/env.sh
# Recreate the environment from the existing dependency lock, without retaining
# duplicate wheel caches on this instance's 32 GB filesystem.
/usr/local/bin/uv sync --project "$REBUTTAL_ROOT/control" --python 3.11 \
  --locked --no-install-project --no-cache
"$REBUTTAL_ROOT/envs/qwen/bin/python" - <<'PY'
import importlib.metadata as m
import json, os
from pathlib import Path
import torch
root = Path(os.environ['REBUTTAL_ROOT'])
packages = {name: m.version(name) for name in ['torch', 'torchvision', 'transformers',
    'huggingface-hub', 'accelerate', 'numpy', 'nltk', 'inflect']}
probe = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(probe @ probe).all()
marker = {'status': 'completed', 'backend': 'vast', 'packages': packages,
          'python': str(root / 'envs/qwen/bin/python'), 'gpu': torch.cuda.get_device_name(0),
          'cuda': torch.version.cuda, 'bf16_gpu_operation': 'passed'}
(root / 'state/setup.json').write_text(json.dumps(marker, indent=2) + '\n')
print(json.dumps(marker, indent=2))
PY
