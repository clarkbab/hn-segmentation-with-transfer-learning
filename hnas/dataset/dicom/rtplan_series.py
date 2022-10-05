import pandas as pd
from typing import List, Optional

from .dicom_file import SOPInstanceUID
from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID
from .region_map import RegionMap
from .rtplan import RTPLAN

class RTPLANSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: SeriesInstanceUID) -> None:
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == DICOMModality.RTPLAN) & (index['series-id'] == id)]
        self.__index = index
        self.__check_index()

    @property
    def default_rtplan(self) -> str:
        if self.__default_rtplan is None:
            self.__load_default_rtplan()
        return self.__default_rtplan

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTPLAN

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_rtplans(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index['sop-id']))

    def rtplan(
        self,
        id: SOPInstanceUID) -> RTPLAN:
        return RTPLAN(self, id)

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTPLANSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_default_rtplan(self) -> None:
        # Preference most recent RTPLAN.
        def_rtplan_id = self.list_rtplans()[-1]
        self.__default_rtplan = self.rtplan(def_rtplan_id)

    def __str__(self) -> str:
        return self.__global_id
