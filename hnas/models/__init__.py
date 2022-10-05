import os
import torch
from typing import Tuple

from hnas import config
from hnas.reporting.models import load_model_manifest
from hnas import types

CHECKPOINT_KEYS = [
    'epoch',
    'global_step'
]

def get_localiser(region: str) -> types.ModelName:
    return (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')

def get_segmenter(
    region: str,
    run: str) -> types.ModelName:
    return (f'segmenter-{region}', run, 'BEST')

def print_checkpoint(model: types.ModelName) -> None:
    # Load data.
    checkpoint = f'{model[2]}.ckpt'
    path = os.path.join(config.directories.models, *model[:2], checkpoint)
    data = torch.load(path, map_location=torch.device('cpu'))

    # Print data.
    for k in CHECKPOINT_KEYS:
        print(f'{k}: {data[k]}')

def replace_checkpoint_alias(
    name: str,
    run: str,
    ckpt: str,
    use_manifest: bool = False) -> Tuple[str, str, str]:
    if ckpt.lower() == 'best': 
        if use_manifest:
            man_df = load_model_manifest()
            ckpts = man_df[(man_df.name == name) & (man_df.run == run) & (man_df.checkpoint != 'last')].sort_values('checkpoint')
            assert len(ckpts) >= 1
            ckpt = ckpts.iloc[-1].checkpoint
        else:
            ckptspath = os.path.join(config.directories.models, name, run)
            ckpts = list(sorted([c.replace('.ckpt', '') for c in os.listdir(ckptspath)]))
            ckpt = ckpts[-1]
    return (name, run, ckpt)
