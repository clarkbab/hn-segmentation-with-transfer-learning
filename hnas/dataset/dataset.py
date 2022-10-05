from enum import Enum

class Dataset:
    @property
    def description(self):
        raise ValueError('Should be overridden')

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    TRAINING = 2
    OTHER = 3

def to_type(name: str) -> DatasetType:
    """
    returns: the DatasetType from string.
    args:
        name: the type string.
    """
    if name.lower() == DatasetType.DICOM.name.lower():
        return DatasetType.DICOM
    elif name.lower() == DatasetType.NIFTI.name.lower():
        return DatasetType.NIFTI
    elif name.lower() == DatasetType.TRAINING.name.lower():
        return DatasetType.TRAINING
    elif name.lower() == DatasetType.OTHER.name.lower():
        return DatasetType.OTHER
    else:
        raise ValueError(f"Dataset type '{name}' not recognised.")
