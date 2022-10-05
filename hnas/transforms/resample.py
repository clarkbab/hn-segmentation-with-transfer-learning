import numpy as np
import SimpleITK as sitk
from typing import Optional

from hnas import types

def resample_3D(
    input: np.ndarray,
    origin: Optional[types.PhysPoint3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None,
    output_origin: Optional[types.PhysPoint3D] = None,
    output_size: Optional[types.ImageSize3D] = None,
    output_spacing: Optional[types.ImageSpacing3D] = None) -> np.ndarray:
    """
    output_origin: 
        - if None, will take on value of 'origin'.
        - if specified, will result in translation of the resulting image (cropping/padding).
    output_size:
        - if None, will take on dimensions of 'input'.
        - if None, will be calculated as a scaling of the 'input' dimensions, where the scaling is determined
            by the ratio of 'spacing' to 'output_spacing'. This ensures, that all image information is preserved
            when doing a spatial resampling.
    output_spacing:
        - if None, will take on value of 'spacing'.
        - if specified, will change the spatial resolution of the image.
    """
    # Convert boolean data to sitk-friendly type.
    boolean = input.dtype == bool
    if boolean:
        input = input.astype('uint8') 

    # Create 'sitk' image and set parameters.
    image = sitk.GetImageFromArray(input)
    if origin is not None:
        image.SetOrigin(tuple(reversed(origin)))
    if spacing is not None:
        image.SetSpacing(tuple(reversed(spacing)))

    # Create resample filter.
    resample = sitk.ResampleImageFilter()
    if boolean:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    if output_origin is not None:
        resample.SetOutputOrigin(tuple(reversed(output_origin)))
    else:
        resample.SetOutputOrigin(image.GetOrigin())
    if output_spacing is not None:
        resample.SetOutputSpacing(tuple(reversed(output_spacing)))
    else:
        resample.SetOutputSpacing(image.GetSpacing())
    if output_size is not None:
        resample.SetSize(tuple(reversed(output_size)))
    else:
        scaling = np.array(image.GetSpacing()) / resample.GetOutputSpacing()
        # Magic formula is: n_new = f * (n - 1) + 1
        size = tuple(int(np.ceil(f * (n - 1) + 1)) for f, n in zip(scaling, image.GetSize()))
        resample.SetSize(size)

    # Perform resampling.
    image = resample.Execute(image)

    # Get output data.
    output = sitk.GetArrayFromImage(image)

    # Convert back to boolean.
    if boolean:
        output = output.astype(bool)

    return output

def resample_box_3D(
    bounding_box: types.Box3D,
    spacing: types.ImageSpacing3D,
    new_spacing: types.ImageSpacing3D) -> types.Box3D:
    """
    returns: a bounding box in resampled coordinates.
    args:
        bounding_box: the bounding box.
        spacing: the current voxel spacing.
        new_spacing: the new voxel spacing.
    """
    # Convert bounding box to label array.
    min, max = bounding_box
    bbox_label = np.zeros(max, dtype=bool)
    slices = tuple(slice(min, max) for min, max in zip(min, max))
    bbox_label[slices] = 1

    # Resample label array.
    bbox_label = resample_3D(bbox_label, spacing, new_spacing)

    # Extract new bounding box.
    non_zero = np.argwhere(bbox_label != 0).astype(int)
    min = tuple(non_zero.min(axis=0))
    max = tuple(non_zero.max(axis=0))
    bounding_box = (min, max)

    return bounding_box
