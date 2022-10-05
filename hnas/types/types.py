import pytorch_lightning as pl
from typing import List, Literal, Sequence, Tuple, Union

Axis = Literal[0, 1, 2]
AxisName = Literal['sagittal', 'coronal', 'axial']
Colour = Union[str, Tuple[float, float, float]]
Crop2D = Tuple[Tuple[int, int], Tuple[int, int]]
Extrema = Literal[0, 1]
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
ModelName = Tuple[str, str, str]
Model = pl.LightningModule
PatientID = Union[int, str]
PatientIDs = Union[Literal['all'], PatientID, Sequence[PatientID]]
PatientView = Literal['axial', 'sagittal', 'coronal'],
PatientRegion = str
PatientRegions = Union[PatientRegion, List[PatientRegion], Literal['all']]
PhysPoint2D = Tuple[float, float]
PhysPoint3D = Tuple[float, float, float]
Point2D = Tuple[int, int]
Point3D = Tuple[int, int, int]
Box2D = Tuple[Point2D, Point2D]
Box3D = Tuple[Point3D, Point3D]
TrainingPartition = Literal['train', 'validation', 'test']
TrainInterval = Union[int, str]