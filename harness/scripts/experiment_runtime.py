"""Explicit execution settings for the VinUni and user-provided Vast hosts."""
import os
from pathlib import Path


ROOT = Path(os.environ.get('REBUTTAL_ROOT', '/mnt/data/quanth/rebuttal'))
CONTROL = ROOT / 'control'


def backend():
    return os.environ.get('REBUTTAL_BACKEND', 'slurm')


def require_execution():
    selected = backend()
    if selected == 'slurm':
        if not os.environ.get('SLURM_JOB_ID'):
            raise SystemExit('Run this stage through Slurm on VinUni.')
    elif selected == 'vast':
        if not os.environ.get('REBUTTAL_ROOT') or not ROOT.is_absolute():
            raise SystemExit('Set an absolute REBUTTAL_ROOT for the Vast instance.')
    else:
        raise SystemExit(f'Unknown execution backend: {selected}')


def execution_id():
    return os.environ.get('SLURM_JOB_ID') or os.environ.get('REBUTTAL_RUN_ID', 'vast4090')


def validate_gpu(torch):
    require_execution()
    if torch.cuda.device_count() != 1:
        raise SystemExit('Expose exactly one GPU for this experiment.')
    device = torch.cuda.get_device_properties(0)
    if backend() == 'slurm':
        if not 30 * 1024**3 < device.total_memory < 45 * 1024**3:
            raise SystemExit('The VinUni configuration requires one 40 GB MIG.')
    elif '4090' not in device.name or device.total_memory < 22 * 1024**3:
        raise SystemExit('The selected Vast configuration requires an RTX 4090 with 24 GB VRAM.')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('This matched experiment requires GPU BF16 support.')
    return device
