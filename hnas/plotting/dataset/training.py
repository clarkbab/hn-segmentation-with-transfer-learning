from typing import Optional, Tuple

from hnas import dataset as ds
from hnas.prediction.dataset.training import get_sample_localiser_prediction
from hnas import types

from ..plotter import plot_distribution, plot_localiser_prediction, plot_regions

def plot_sample_regions(
    dataset: str,
    sample_idx: int,
    regions: types.PatientRegions = 'all',
    **kwargs) -> None:
    # Load data.
    sample = ds.get(dataset, 'training').sample(sample_idx)
    input = sample.input
    labels = sample.label(region=regions)
    spacing = sample.spacing
    
    # Plot.
    plot_regions(sample_idx, input, labels, spacing, region=regions, **kwargs)

def plot_sample_localiser_prediction(
    dataset: str,
    sample_idx: str,
    region: str,
    localiser: types.ModelName,
    **kwargs) -> None:
    # Load data.
    sample = ds.get(dataset, 'training').sample(sample_idx)
    input = sample.input
    label = sample.label(region=region)[region]
    spacing = sample.spacing

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    # Make prediction.
    pred = get_sample_localiser_prediction(dataset, sample_idx, localiser, truncate=truncate)
    
    # Plot.
    plot_localiser_prediction(sample_idx, region, input, label, spacing, pred, **kwargs)

def plot_sample_distribution(
    dataset: str,
    sample_idx: int,
    figsize: Tuple[float, float] = (12, 6),
    range: Optional[Tuple[float, float]] = None,
    resolution: float = 10) -> None:
    # Load data.
    input = ds.get(dataset, 'training').sample(sample_idx).input
    
    # Plot distribution.
    plot_distribution(input, figsize=figsize, range=range, resolution=resolution)
