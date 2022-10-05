from enum import Enum

SeriesInstanceUID = str

class DICOMModality(str, Enum):
    CT = 'CT'
    RTSTRUCT = 'RTSTRUCT'
    RTPLAN = 'RTPLAN'
    RTDOSE = 'RTDOSE'

# Abstract class.
class DICOMSeries:
    @property
    def modality(self) -> DICOMModality:
        raise NotImplementedError("Child class must implement 'modality'.")
