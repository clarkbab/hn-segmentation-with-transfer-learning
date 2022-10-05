import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset

from .ct_series import CTSeries
from .dicom_file import DICOMFile, SOPInstanceUID

class RTPLAN(DICOMFile):
    def __init__(
        self,
        series: 'RTPLANSeries',
        id: SOPInstanceUID):
        self.__global_id = f"{series} - {id}"
        self.__id = id
        self.__ref_ct = None        # Lazy-loaded.
        self.__series = series

        # Get index.
        index = self.__series.index
        index = index[index['sop-id'] == self.__id]
        self.__index = index
        self.__check_index()
        self.__path = self.__index.iloc[0]['filepath']

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    @property
    def series(self) -> str:
        return self.__series

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def get_rtplan(self) -> FileDataset:
        return dcm.read_file(self.__path)

    def __str__(self) -> str:
        return self.__global_id

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTPLAN '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTPLANs found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")
