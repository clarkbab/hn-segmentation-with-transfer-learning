import os
import shutil
from typing import List

from hnas import config

from .nifti_dataset import NIFTIDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'nifti')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    os.makedirs(ds_path)
    return NIFTIDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
