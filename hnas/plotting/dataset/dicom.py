from typing import List, Union

from ..plotter import plot_regions, plot_segmenter_prediction
from hnas import dataset as ds
from hnas.prediction.dataset.dicom import load_segmenter_predictions
from hnas import types

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    regions: types.PatientRegions = 'all',
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'dicom').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(region=regions)
    spacing = patient.ct_spacing
    
    # Plot.
    plot_regions(pat_id, ct_data, region_data, spacing, region=regions, **kwargs)

def plot_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    models: Union[str, List[str]],
    show_dose: bool = False,
    **kwargs) -> None:
    if type(models) == str:
        models = [models]
    
    # Load data.
    patient = ds.get(dataset, 'dicom').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(region=region)[region]
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None

    # Load model predictions.
    preds = []
    for model in models:
        pred = load_segmenter_predictions(dataset, pat_id, model, region)
        preds.append(pred)

    # Plot.
    plot_segmenter_prediction(pat_id, region, ct_data, region_data, spacing, preds, dose_data=dose_data, pred_labels=models, **kwargs)
