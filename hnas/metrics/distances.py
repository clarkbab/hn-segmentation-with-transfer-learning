import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Literal, Tuple, Union

from hnas.geometry import get_extent, get_extent_centre
from hnas import types

def apl(
    surf_dists: Tuple[np.ndarray, np.array],
    spacing: types.ImageSpacing3D,
    tol: float,
    unit: Literal['mm', 'voxels'] = 'mm') -> float:
    b_to_a_surf_dists = surf_dists[1]   # Only look at dists from 'GT' back to 'pred'. 
    b_to_a_overlap = b_to_a_surf_dists[b_to_a_surf_dists > tol]
    assert spacing[0] == spacing[1], f"In-plane spacing should be equal when calculating APL, got '{spacing}'."
    if unit == 'mm':
        return len(b_to_a_overlap) * spacing[0]
    else:
        return len(b_to_a_overlap)

def hausdorff_distance(
    surf_dists: Tuple[np.ndarray, np.ndarray],
    p: float = 100) -> float:
    return np.max((np.percentile(surf_dists[0], p), np.percentile(surf_dists[1], p)))

def mean_surface_distance(
    surf_dists: Tuple[np.ndarray, np.ndarray]) -> float:
    return np.mean((np.mean(surf_dists[0]), np.mean(surf_dists[1])))

def surface_dice(
    surf_dists: Tuple[np.ndarray, np.ndarray],
    tol: float) -> float:
    a_to_b_overlap = surf_dists[0][surf_dists[0] <= tol]
    b_to_a_overlap = surf_dists[1][surf_dists[1] <= tol]
    return (len(a_to_b_overlap) + len(b_to_a_overlap)) / (len(surf_dists[0]) + len(surf_dists[1]))

def all_distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    tols: Union[int, float, List[Union[int, float]]]) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"Metric 'distances' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")
    if type(tols) == int or type(tols) == float:
        tols = [tols]

    # Add metrics.
    surf_dists = surface_distances(a, b, spacing)
    metrics = {
        'hd': hausdorff_distance(surf_dists),
        'hd-95': hausdorff_distance(surf_dists, 95),
        'msd': mean_surface_distance(surf_dists)
    }
    
    # Add metrics with 'tolerances'.
    for tol in tols:
        metrics[f'apl-mm-tol-{tol}'] = apl(surf_dists, spacing, tol, unit='mm')
        metrics[f'apl-voxel-tol-{tol}'] = apl(surf_dists, spacing, tol, unit='voxel')
        metrics[f'surface-dice-tol-{tol}'] = surface_dice(surf_dists, tol)

    return metrics

def surface_distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"Metric 'distances' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Convert to SimpleITK images.
    a_itk = sitk.GetImageFromArray(a.astype('uint8'))
    a_itk.SetSpacing(tuple(reversed(spacing)))
    b_itk = sitk.GetImageFromArray(b.astype('uint8'))
    b_itk.SetSpacing(tuple(reversed(spacing)))

    # Get surface voxels.
    a_surface = sitk.LabelContour(a_itk, False)
    b_surface = sitk.LabelContour(b_itk, False)

    # Compute distance maps - calculate 'absolute' distances as direction of deviation from other surface doesn't matter.
    a_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(a_surface, useImageSpacing=True, squaredDistance=False, insideIsPositive=False))
    b_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(b_surface, useImageSpacing=True, squaredDistance=False, insideIsPositive=False))

    # Convert 'sitk' images to 'numpy' arrays.
    a_dist_map = sitk.GetArrayFromImage(a_dist_map)
    b_dist_map = sitk.GetArrayFromImage(b_dist_map)
    a_surface = sitk.GetArrayFromImage(a_surface)
    b_surface = sitk.GetArrayFromImage(b_surface)

    # Get voxel/surface min distances.
    a_to_b_surface_min_dists = b_dist_map[a_surface == 1]
    b_to_a_surface_min_dists = a_dist_map[b_surface == 1]

    return a_to_b_surface_min_dists, b_to_a_surface_min_dists

def batch_mean_all_distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    tols: Union[float, List[float]]) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'batch_mean_all_distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"Metric 'batch_mean_all_distances' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Average metrics over all batch items.
    mean_dists = {}
    for a, b in zip(a, b):
        dists = all_distances(a, b, spacing, tols)
        for metric, value in dists.items():
            if metric not in mean_dists:
                dists[metric] = []
            dists[metric].append(value)
    mean_dists = dict((metric, np.mean(values)) for metric, values in mean_dists.items())
    return mean_dists

def extent_centre_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent centres for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_centre_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"Metric 'extent_centre_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_centre_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extent centres.
    a_cent = get_extent_centre(a)
    b_cent = get_extent_centre(b)

    # Get distance between centres.
    dists = np.abs(np.array(b_cent) - np.array(a_cent))    
    dists_mm = spacing * dists
    return dists_mm

def extent_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent boundaries for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"Metric 'extent_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = get_extent(a)
    b_ext = get_extent(b)

    # Calculate distances.
    a = np.array(a_ext)
    a[1] = -a[1]
    b = np.array(b_ext)
    b[1] = -b[1]
    dists = np.max(a - b, axis=0)
    dists_mm = spacing * dists
    return dists_mm

def get_encaps_dist_vox(
    a: np.ndarray,
    b: np.ndarray) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_vox' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"'get_encaps_dist_vox' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_vox' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = get_extent(a)
    b_ext = get_extent(b)

    # Calculate distances.
    a = np.array(a_ext)
    a[1] = -a[1]
    b = np.array(b_ext)
    b[1] = -b[1]
    dist = np.max(a - b, axis=0)
    return dist

def get_encaps_dist_mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_mm' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"'get_encaps_dist_mm' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_mm' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    dist = get_encaps_dist_vox(a, b)
    dist_mm = tuple(np.array(spacing) * dist)
    return dist_mm
