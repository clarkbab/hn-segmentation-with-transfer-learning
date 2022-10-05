import os
import shutil
from typing import List

from hnas import config

from .training_dataset import TrainingDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'training')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def get(name: str) -> TrainingDataset:
    if exists(name):
        return TrainingDataset(name)
    else:
        raise ValueError(f"TrainingDataset '{name}' doesn't exist.")

def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    return os.path.exists(ds_path)

def create(name: str) -> TrainingDataset:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    os.makedirs(ds_path)
    return TrainingDataset(name, check_processed=False)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
