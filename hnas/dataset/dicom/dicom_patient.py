import pandas as pd
from typing import Any, List, Optional

from hnas import types
from hnas.utils import append_row

from .dicom_study import DICOMStudy
from .region_map import RegionMap
from .rtdose_series import RTDOSESeries
from .rtplan_series import RTPLANSeries
from .rtstruct_series import RTSTRUCTSeries

class DICOMPatient:
    def __init__(
        self,
        dataset: 'DICOMDataset',
        id: types.PatientID,
        ct_from: Optional['DICOMPatient'] = None,
        region_dups: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None,
        trimmed: bool = False):
        if trimmed:
            self.__global_id = f"{dataset} - {id} (trimmed)"
        else:
            self.__global_id = f"{dataset} - {id}"
        self.__ct_from = ct_from
        self.__default_rtdose = None        # Lazy-loaded.
        self.__default_rtplan = None        # Lazy-loaded.
        self.__default_rtstruct = None      # Lazy-loaded.
        self.__default_study = None         # Lazy-loaded.
        self.__dataset = dataset
        self.__id = str(id)
        self.__region_dups = region_dups
        self.__region_map = region_map

        # Get patient index.
        index = self.__dataset.index
        index = index[index['patient-id'] == str(id)]
        self.__index = index

        # Check that patient ID exists.
        if len(index) == 0:
            raise ValueError(f"Patient '{self}' not found in index for dataset '{dataset}'.")

    @property
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate

    @property
    def ct_data(self):
        return self.default_rtstruct.ref_ct.data

    @property
    def ct_from(self) -> str:
        return self.__ct_from

    @property
    def ct_offset(self):
        return self.default_rtstruct.ref_ct.offset

    @property
    def ct_size(self):
        return self.default_rtstruct.ref_ct.size

    @property
    def ct_spacing(self):
        return self.default_rtstruct.ref_ct.spacing

    @property
    def dataset(self) -> str:
        return self.__dataset

    @property
    def default_rtdose(self) -> str:
        if self.__default_rtdose is None:
            self.__load_default_rtdose_and_rtplan()
        return self.__default_rtdose

    @property
    def default_rtplan(self) -> RTPLANSeries:
        if self.__default_rtplan is None:
            self.__load_default_rtdose_and_rtplan()
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> RTSTRUCTSeries:
        if self.__default_rtstruct is None:
            self.__load_default_rtstruct()
        return self.__default_rtstruct
    
    @property
    def default_study(self) -> DICOMStudy:
        if self.__default_study is None:
            self.__load_default_study()
        return self.__default_study

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dose_data(self):
        return self.default_rtdose.data

    @property
    def dose_offset(self):
        return self.default_rtdose.offset

    @property
    def dose_size(self):
        return self.default_rtdose.size

    @property
    def dose_spacing(self):
        return self.default_rtdose.spacing

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def id(self) -> str:
        return self.__id

    @property
    def name(self) -> str:
        return self.get_cts()[0].PatientName

    @property
    def region_dups(self) -> List[str]:
        return self.__region_dups

    @property
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

    def ct_slice_summary(self, *args, **kwargs):
        return self.default_rtstruct.ref_ct.slice_summary(*args, **kwargs)

    def ct_summary(self, *args, **kwargs):
        return self.default_rtstruct.ref_ct.summary(*args, **kwargs)

    def get_rtdose(self, *args, **kwargs):
        return self.__default_rtdose_series.get_rtdose(*args, **kwargs)

    def get_cts(self, *args, **kwargs):
        return self.default_rtstruct.ref_ct.get_cts(*args, **kwargs)
 
    def get_rtstruct(self, *args, **kwargs):
        return self.default_rtstruct.get_rtstruct(*args, **kwargs)

    def has_region(self, *args, **kwargs):
        return self.default_rtstruct.has_region(*args, **kwargs)

    def info(self) -> pd.DataFrame:
        # Define dataframe structure.
        cols = {
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

        # Add row.
        df = append_row(df, data)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    def list_studies(self) -> List[str]:
        return list(sorted(self.__index['study-id'].unique()))

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    def region_summary(self, *args, **kwargs):
        return self.default_rtstruct.region_summary(*args, **kwargs)

    def study(
        self,
        id: str) -> DICOMStudy:
        return DICOMStudy(self, id, region_dups=self.__region_dups, region_map=self.__region_map)

    def __load_default_rtdose_and_rtplan(self) -> None:
        self.__default_rtplan = self.default_study.default_rtplan
        self.__default_rtdose = self.default_study.default_rtdose

    def __load_default_rtstruct(self) -> None:
        self.__default_rtstruct = self.default_study.default_rtstruct

    def __load_default_study(self) -> None:
        # Preference the most recent study.
        def_study_id = self.list_studies()[-1]
        self.__default_study = self.study(def_study_id)

    def __str__(self) -> str:
        return self.__global_id
