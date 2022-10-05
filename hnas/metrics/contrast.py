import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SimpleITK as sitk
from typing import List, Tuple

from hnas import types

def __contrast_boundary(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    d: float = 5) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'contrast' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.float or b.dtype != np.bool_:
        raise ValueError(f"Metric 'contrast' expects (float, boolean) arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if b.sum() == 0:
        raise ValueError(f"Metric 'contrast' can't be calculated on empty 'b' set. Got cardinalities '{b.sum()}'.")

    # Get distance map.
    b_itk = b.astype('uint8')
    b_itk = sitk.GetImageFromArray(b_itk)
    b_itk.SetSpacing(tuple(reversed(spacing)))
    b_dist_map = sitk.SignedMaurerDistanceMap(b_itk, useImageSpacing=True, squaredDistance=False, insideIsPositive=False)
    b_dist_map = sitk.GetArrayFromImage(b_dist_map)

    # Get mean boundary intensity.
    mask = np.zeros_like(a, dtype=bool)
    mask[(b_dist_map > 0) & (b_dist_map <= d)] = 1

    return mask

def contrast(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    d: float = 5) -> Tuple[List[float], List[float]]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'contrast' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.float or b.dtype != np.bool_:
        raise ValueError(f"Metric 'contrast' expects (float, boolean) arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if b.sum() == 0:
        raise ValueError(f"Metric 'contrast' can't be calculated on empty 'b' set. Got cardinalities '{b.sum()}'.")
    
    # Get diff between mean values.
    boundary_mask = __contrast_boundary(a, b, spacing, d=d)
    b_values = a[b == 1]
    boundary_values = a[boundary_mask == 1]
    contrast = np.abs(boundary_values.mean() - b_values.mean())

    return contrast

def plot_contrast(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    slice_idx: int,
    d: float = 5) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'contrast' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.float or b.dtype != np.bool_:
        raise ValueError(f"Metric 'contrast' expects (float, boolean) arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if b.sum() == 0:
        raise ValueError(f"Metric 'contrast' can't be calculated on empty 'b' set. Got cardinalities '{b.sum()}'.")

    # Plot patient.
    boundary_mask = __contrast_boundary(a, b, spacing, d=d)
    inv_boundary_mask = np.zeros_like
    plt.figure(figsize=(12, 12))
    plt.imshow(np.tranpose(a)[i, :, :])
    plt.imshow(np.transpose())

def plot_contrast_hist(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    d: float = 5) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'contrast' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.float or b.dtype != np.bool_:
        raise ValueError(f"Metric 'contrast' expects (float, boolean) arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if b.sum() == 0:
        raise ValueError(f"Metric 'contrast' can't be calculated on empty 'b' set. Got cardinalities '{b.sum()}'.")

    # Plot histograms.
    boundary_mask = __contrast_boundary(a, b, spacing, d=d)
    b_values = a[b == 1]
    boundary_values = a[boundary_mask == 1]
    colours = sns.color_palette('colorblind')[:2]
    plt.figure(figsize=(12, 6))
    plt.hist(b_values, color=colours[0], alpha=0.5)
    plt.hist(boundary_values, color=colours[1], alpha=0.5)
    plt.axvline(b_values.mean(), color=colours[0], linestyle='--')
    plt.axvline(boundary_values.mean(), color=colours[1], linestyle='--')
    plt.show()
