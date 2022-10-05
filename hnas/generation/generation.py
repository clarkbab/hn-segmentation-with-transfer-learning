import numpy as np
from typing import Optional, Union

from hnas import types

def create_cylinder(
    img_size: types.ImageSize3D,
    centre: Optional[types.PhysPoint3D] = None,
    height: Optional[float] = None,
    radius: Optional[float] = None) -> np.ndarray:
    img_size_2D = img_size[:-1]

    # Default to middle of image.
    if centre is None:
        centre = tuple((s - 1) / 2 for s in img_size)
    centre_2D = centre[:-1]
    # Default to largest radius.
    if radius is None:
        radius = np.min((*centre, *(np.array(img_size) - centre)))
    

    points_2D = np.ogrid[list(slice(None, s) for s in img_size_2D)]
    dists_from_centre_2D = np.sqrt(sum((p - c) ** 2 for p, c in zip(points_2D, centre_2D)))
    img = np.zeros(img_size, dtype=bool)
    min_z = int(np.ceil(centre[2] - ((height - 1) / 2)))
    max_z = int(np.floor(centre[2] + ((height - 1) / 2)))
    for i in range(min_z, max_z + 1):
        img[:, :, i] = dists_from_centre_2D <= radius
    return img

def create_n_sphere(
    img_size: Union[types.ImageSize2D, types.ImageSize3D],
    centre: Optional[Union[types.PhysPoint2D, types.PhysPoint3D]] = None,
    radius: Optional[float] = None) -> np.ndarray:
    # Default to middle of image.
    if centre is None:
        centre = tuple((s - 1) / 2 for s in img_size)
    # Default to largest radius.
    if radius is None:
        radius = np.min((*centre, *(np.array(img_size) - centre)))
    points = np.ogrid[list(slice(None, s) for s in img_size)]
    dists_from_centre = np.sqrt(sum((p - c) ** 2 for p, c in zip(points, centre)))
    img = np.zeros(img_size, dtype=bool)
    img[dists_from_centre <= radius] = 1
    return img
