import numpy as np
from typing import Optional, Tuple, Union

from hnas import types

def get_extent(a: np.ndarray) -> Optional[Union[types.Box2D, types.Box3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'get_extent' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    if a.sum() > 0:
        non_zero = np.argwhere(a != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)
    else:
        box = None

    return box

def get_extent_centre(a: np.ndarray) -> Optional[Union[types.Point2D, types.Point3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'get_extent_centre' expected a boolean array, got '{a.dtype}'.")

    # Get extent.
    extent = get_extent(a)

    if extent:
        # Find the extent centre.
        centre = tuple(np.floor(np.array(extent).sum(axis=0) / 2).astype(int))
    else:
        return None

    return centre

def get_extent_width_vox(a: np.ndarray) -> Optional[Union[types.ImageSize2D, types.ImageSize3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'get_extent_width_vox' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    extent = get_extent(a)
    if extent:
        min, max = extent
        width = tuple(np.array(max) - min)
        return width
    else:
        return None

def get_extent_width_mm(
    a: np.ndarray,
    spacing: Tuple[float, float, float]) -> Optional[Union[Tuple[float, float], Tuple[float, float, float]]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'get_extent_width_mm' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent in mm.
    ext_width_vox = get_extent_width_vox(a)
    if ext_width_vox is None:
        return None
    ext_width = tuple(np.array(ext_width_vox) * spacing)
    return ext_width
