import numpy as np
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset

from hnas.transforms import resample_3D
from hnas.types import ImageSize3D, ImageSpacing3D, PhysPoint3D

from .dicom_file import DICOMFile, SOPInstanceUID

class RTDOSE(DICOMFile):
    def __init__(
        self,
        series: 'RTDOSESeries',
        id: SOPInstanceUID):
        self.__data = None      # Lazy-loaded.
        self.__global_id = f"{series} - {id}"
        self.__offset = None    # Lazy-loaded.
        self.__id = id
        self.__series = series
        self.__spacing = None   # Lazy-loaded.

        # Get index.
        index = self.__series.index
        index = index[index['sop-id'] == self.__id]
        self.__index = index
        self.__check_index()
        self.__path = self.__index.iloc[0]['filepath']

    @property
    def data(self) -> np.ndarray:
        if self.__data is None:
            self.__load_rtdose_data()
        return self.__data

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
    def offset(self) -> PhysPoint3D:
        if self.__offset is None:
            self.__load_rtdose_data()
        return self.__offset

    @property
    def path(self) -> str:
        return self.__path

    @property
    def series(self) -> str:
        return self.__series

    @property
    def size(self) -> ImageSize3D:
        return self.data.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        if self.__spacing is None:
            self.__load_rtdose_data()
        return self.__spacing

    def get_rtdose(self) -> FileDataset:
        return dcm.read_file(self.__path)

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTPLAN '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTPLANs found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __load_rtdose_data(self) -> None:
        rtdose = self.get_rtdose()

        # Store offset.
        offset = rtdose.ImagePositionPatient
        self.__offset = tuple(int(s) for s in offset)

        # Store spacing.
        spacing_x_y = rtdose.PixelSpacing 
        z_diffs = np.unique(np.diff(rtdose.GridFrameOffsetVector))
        assert len(z_diffs) == 1
        spacing_z = z_diffs[0]
        self.__spacing = tuple(np.append(spacing_x_y, spacing_z))

        # Store dose data.
        pat = self.__series.study.patient
        data = np.transpose(rtdose.pixel_array)
        data = rtdose.DoseGridScaling * data
        self.__data = resample_3D(data, origin=self.__offset, spacing=self.__spacing, output_origin=pat.ct_offset, output_size=pat.ct_size, output_spacing=pat.ct_spacing) 

    def __str__(self) -> str:
        return self.__global_id
