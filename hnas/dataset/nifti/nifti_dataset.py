import os
import pandas as pd
from typing import List, Literal, Optional, Union

from hnas import config
from hnas import logging
from hnas import types

from ..dataset import Dataset, DatasetType
from .nifti_patient import NIFTIPatient

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self.__global_id = f"NIFTI: {name}"
        self.__anon_index = None                # Lazy-loaded.
        self.__excluded_labels = None          # Lazy-loaded.
        self.__group_index = None               # Lazy-loaded.
        self.__loaded_anon_index = False
        self.__loaded_excluded_labels = False
        self.__loaded_group_index = False
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'nifti', name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @property
    def anon_index(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_anon_index:
            self.__load_anon_index()
            self.__loaded_anon_index = True
        return self.__anon_index
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def excluded_labels(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_excluded_labels:
            self.__load_excluded_labels()
            self.__loaded_excluded_labels = True
        return self.__excluded_labels

    @property
    def group_index(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_group_index:
            self.__load_group_index()
            self.__loaded_group_index = True
        return self.__group_index

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    def list_patients(
        self,
        labels: Literal['included', 'excluded', 'all'] = 'included',
        region: types.PatientRegions = 'all') -> List[str]:

        # Load patients.
        ct_path = os.path.join(self.__path, 'data', 'ct')
        files = list(sorted(os.listdir(ct_path)))
        pat_ids = [f.replace('.nii.gz', '') for f in files]

        # Filter by 'region'.
        pat_ids = list(filter(lambda pat_id: self.patient(pat_id).has_region(region, labels=labels), pat_ids))
        return pat_ids

    def patient(
        self,
        id: Union[int, str]) -> NIFTIPatient:
        return NIFTIPatient(self, id, excluded_labels=self.excluded_labels)
    
    def __load_anon_index(self) -> None:
        filepath = os.path.join(self.__path, 'anon-index.csv')
        if os.path.exists(filepath):
            self.__anon_index = pd.read_csv(filepath).astype({ 'anon-id': str, 'origin-patient-id': str })
        else:
            self.__anon_index = None
    
    def __load_excluded_labels(self) -> None:
        filepath = os.path.join(self.__path, 'excluded-labels.csv')
        if os.path.exists(filepath):
            self.__excluded_labels = pd.read_csv(filepath).astype({ 'patient-id': str })
            self.__excluded_labels = self.__excluded_labels.sort_values(['patient-id', 'region'])

            # Drop duplicates.
            dup_cols = ['patient-id', 'region']
            dup_df = self.__excluded_labels[self.__excluded_labels[dup_cols].duplicated()]
            if len(dup_df) > 0:
                logging.warning(f"Found {len(dup_df)} duplicate entries in 'excluded-labels.csv', removing.")
                self.__excluded_labels = self.__excluded_labels[~self.__excluded_labels[dup_cols].duplicated()]
        else:
            self.__excluded_labels = None

    def __load_group_index(self) -> None:
        filepath = os.path.join(self.__path, 'group-index.csv')
        if os.path.exists(filepath):
            self.__group_index = pd.read_csv(filepath).astype({ 'patient-id': str })
        else:
            self.__group_index = None

    def __str__(self) -> str:
        return self.__global_id
    