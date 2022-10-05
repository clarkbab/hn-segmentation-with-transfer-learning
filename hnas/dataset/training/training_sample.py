from numpy.lib.arraysetops import intersect1d
from hnas.types.types import PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Union

from hnas.regions import region_to_list
from hnas import types
from hnas.utils import arg_to_list

class TrainingSample:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        id: Union[int, str]):
        self.__dataset = dataset
        self.__id = int(id)
        self.__index = None         # Lazy-loaded.
        self.__global_id = f'{self.__dataset} - {self.__id}'
        self.__group_id = None      # Lazy-loaded.
        self.__spacing = self.__dataset.params['output-spacing']

        # Load sample index.
        if self.__id not in self.__dataset.list_samples():
            raise ValueError(f"Sample '{self.__id}' not found for dataset '{self.__dataset}'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def group_id(self) -> str:
        if self.__group_id is None:
            self.__group_id = self.index.iloc[0]['group-id']
        return self.__group_id

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> str:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def spacing(self) -> types.ImageSpacing3D:
        return self.__spacing

    def list_regions(
        self,
        include_empty: bool = False) -> List[str]:
        # Don't list 'empty' regions.
        df = self.index
        if 'empty' in df and not include_empty:
            df = df[~df['empty']]
        return list(sorted(df.region))

    def has_region(
        self,
        region: str) -> bool:
        return region in self.list_regions()

    @property
    def origin(self) -> Tuple:
        idx = self.__dataset.index
        record = idx[idx['sample-id'] == self.__id].iloc[0]
        return (record['origin-dataset'], record['origin-patient-id'])

    @property
    def input(self) -> np.ndarray:
        # Load the input data.
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}.npz')
        data = np.load(filepath)['data']
        data = data.astype(np.float32)
        return data

    def label(
        self,
        region: types.PatientRegions = 'all') -> Dict[str, np.ndarray]:
        regions = arg_to_list(region, str, literals={ 'all': self.list_regions() })

        # Load the label data.
        data = {}
        for region in regions:
            filepath = os.path.join(self.__dataset.path, 'data', 'labels', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Region '{region}' not found for sample '{self}'.")
            label = np.load(filepath)['data']
            data[region] = label

        return data

    def pair(
        self,
        region: types.PatientRegions = 'all') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.input, self.label(region=region)

    def __load_index(self) -> None:
        index = self.__dataset.index
        index = index[index['sample-id'] == self.__id]
        assert len(index == 1)
        self.__index = index

    def __str__(self) -> str:
        return self.__global_id
