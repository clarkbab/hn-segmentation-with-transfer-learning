import numpy as np
from typing import Optional, Tuple, Union

from hnas.geometry import get_box
from hnas import types

def crop_or_pad_2D(
    data: np.ndarray,
    bounding_box: types.Box2D,
    fill: float = 0) -> np.ndarray:
    # Convert args to 3D.
    data = np.expand_dims(data, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D pad code.
    data = crop_or_pad_3D(data, bounding_box, fill=fill)

    # Remove final dimension.
    data = np.squeeze(data, axis=2)

    return data

def crop_or_pad_3D(
    data: np.ndarray,
    bounding_box: types.Box3D,
    fill: float = 0) -> np.ndarray:
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."

    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Perform padding.
    size = np.array(data.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))
    data = np.pad(data, padding, constant_values=fill)

    # Perform cropping.
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, data.shape))
    data = data[slices]

    return data

def crop_foreground_3D(
    data: np.ndarray,
    crop: types.Box3D) -> np.ndarray:
    cropped = np.zeros_like(data).astype(bool)
    slices = tuple(slice(min, max) for min, max in zip(*crop))
    cropped[slices] = data[slices]
    return cropped

def centre_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    fill: float = 0) -> np.ndarray:
    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def top_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    fill: float = 0) -> np.ndarray:
    # Centre crop x/y axes.
    to_crop = data.shape[:2] - np.array(size[:2])
    xy_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    xy_max = xy_min + size[:2]

    # Top crop z axis to maintain HN region.
    z_max = data.shape[2]
    z_min = z_max - size[2]

    # Perform crop or padding.
    bounding_box = ((*xy_min, z_min), (*xy_max, z_max)) 
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def point_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    point: types.Point3D,
    fill: float = 0,
    return_box: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, types.Box3D]]:
    # Perform the crop or pad.
    box = get_box(point, size)
    data = crop_or_pad_3D(data, box, fill=fill)

    if return_box:
        return (data, box)
    else:
        return data

def crop_or_pad_point(
    point: Union[types.Point2D, types.Point3D],
    crop: Union[types.Box2D, types.Box3D]) -> Optional[Union[types.Point2D, types.Point3D]]:
    # Check dimensions.
    assert len(point) == len(crop[0]) and len(point) == len(crop[1])

    crop = np.array(crop)
    point = np.array(point).reshape(1, crop.shape[1])

    # Get decision variables.
    decisions = np.stack((point >= crop[0], point < crop[1]), axis=0)

    # Check if point is in crop window.
    if np.all(decisions):
        point -= crop[0]
        point = tuple(point.flatten())
    else:
        point = None

    return point

def crop_or_pad_box(
    box: Union[types.Box2D, types.Box3D],
    crop: Union[types.Box2D, types.Box3D]) -> Optional[Union[types.Box2D, types.Box3D]]:
    box = np.array(box, dtype=int)
    crop = np.array(crop, dtype=int)

    # Get decision variables.
    decisions = np.stack((crop[0] <= box[1], crop[1] > box[0], crop[0] <= box[0], crop[1] > box[1]), axis=0)

    # Check that box is contained in crop.
    if np.all(decisions[0:2]):
        new_box = np.zeros_like(box, dtype=int)

        # Add viable box values.
        idx = np.nonzero(decisions[2:4])
        new_box[idx] = box[idx]

        # Project outside points to crop boundaries.
        idx = np.nonzero(~decisions[2:4])
        new_box[idx] = crop[idx]

        # Crop points.
        new_box[0] = crop_or_pad_point(tuple(new_box[0]), crop)
        new_box[1] = crop_or_pad_point(tuple(new_box[1]), crop)

        # Convert to tuple.
        new_box = tuple(tuple(p) for p in new_box)
    else:
        new_box = None

    return new_box
