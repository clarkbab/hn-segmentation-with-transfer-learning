import numpy as np
import os
from typing import Union

from hnas import config

class OtherSample:
    def __init__(
        self,
        dataset: 'OtherDataset',
        id: Union[int, str]):
        self._global_id = f'{dataset} - {id}'
        self._id = id
        self._path = os.path.join(config.directories.datasets, 'other', dataset.name, 'data', f'{id}.npz')
        if not os.path.exists(self._path):
            raise ValueError(f"OtherSample '{self}' not found.")
    
    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id
    
    @property
    def path(self) -> str:
        return self._path

    def data(self) -> np.ndarray:
        return np.load(self._path)['data']
