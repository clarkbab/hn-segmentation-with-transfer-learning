import os
from typing import List, Union

from hnas import config

from ..dataset import Dataset, DatasetType
from .other_sample import OtherSample

class OtherDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._global_id = f"OTHER: {name}"
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'other', name)
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")
    
    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id
    
    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> DatasetType:
        return DatasetType.OTHER

    def list_samples(self) -> List[int]:
        datapath = os.path.join(self._path, 'data')
        files = os.listdir(datapath)
        samples = list(sorted([int(f.replace('.npz', '')) for f in files]))
        return samples

    def sample(
        self,
        id: Union[int, str]) -> OtherSample:
        return OtherSample(self, id)

