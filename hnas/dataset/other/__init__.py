import os
import shutil
from typing import List

from hnas import config

from .other_dataset import OtherDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'other')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'other', name)
    os.makedirs(ds_path)
    return OtherDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'other', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
