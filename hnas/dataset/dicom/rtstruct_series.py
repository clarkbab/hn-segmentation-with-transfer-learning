import pandas as pd
from typing import List, Optional

from hnas.types.types import PatientRegion

from .dicom_file import SOPInstanceUID
from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID
from .region_map import RegionMap
from .rtstruct import RTSTRUCT

class RTSTRUCTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: SeriesInstanceUID,
        region_dups: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__default_rtstruct = None      # Lazy-loaded.
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__region_dups = region_dups
        self.__region_map = region_map
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == DICOMModality.RTSTRUCT) & (index['series-id'] == id)]
        self.__index = index
        self.__check_index()

    @property
    def default_rtstruct(self) -> RTSTRUCT:
        if self.__default_rtstruct is None:
            self.__load_default_rtstruct()
        return self.__default_rtstruct

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTSTRUCT

    @property
    def study(self) -> str:
        return self.__study

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def list_rtstructs(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index['sop-id']))

    def rtstruct(
        self,
        id: SOPInstanceUID) -> RTSTRUCT:
        return RTSTRUCT(self, id, region_dups=self.__region_dups, region_map=self.__region_map)

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTSTRUCTSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_default_rtstruct(self) -> None:
        # Preference most recent RTSTRUCT.
        def_rtstruct_id = self.list_rtstructs()[-1]
        self.__default_rtstruct = self.rtstruct(def_rtstruct_id)

    def __str__(self) -> str:
        return self.__global_id
