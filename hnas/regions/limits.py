import numpy as np

from hnas.geometry import get_extent, get_extent_width_mm
from hnas import logging
from hnas.transforms import crop_foreground_3D
from hnas import types

# Limits in mm.
class RegionLimits:
    SpinalCord = (-1, -1, 290)  # Assuming first-stage has spacing (4, 4, 4).

def truncate_spine(
    pred: np.ndarray,
    spacing: types.ImageSpacing3D) -> np.ndarray:
    ext_width = get_extent_width_mm(pred, spacing)
    if ext_width is not None and ext_width[2] > RegionLimits.SpinalCord[2]:
        # Crop caudal end of spine.
        logging.info(f"Truncating caudal end of 'SpinalCord'. Got length (z-axis) of '{ext_width[2]}mm', maximum is '{RegionLimits.SpinalCord[2]}mm'.")
        top_z = get_extent(pred)[1][2]
        bottom_z = int(np.ceil(top_z - RegionLimits.SpinalCord[2] / spacing[2]))
        crop = ((0, 0, bottom_z), tuple(np.array(pred.shape) - 1))
        pred = crop_foreground_3D(pred, crop)

    return pred
