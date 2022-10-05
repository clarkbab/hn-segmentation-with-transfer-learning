import re
from typing import Dict, List, Optional, Union

from hnas import dataset as ds
from hnas.geometry import get_extent_centre
from hnas import logging
from hnas.prediction.dataset.nifti import create_patient_localiser_prediction, create_patient_segmenter_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from hnas import types

from ..plotter import plot_localiser_prediction, plot_regions, plot_segmenter_prediction

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,     # Gives 'regions' different names to those used for loading the data.
    show_dose: bool = False,
    **kwargs) -> None:
    if type(regions) == str:
        regions = [regions]

    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(region=regions) if regions is not None else None
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None

    if centre_of is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                centre_of = patient.region_data(region=centre_of)[centre_of]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = patient.region_data(region=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]

    # Plot.
    plot_regions(pat_id, ct_data.shape, spacing, centre_of=centre_of, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.ModelName,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    load_prediction: bool = True,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(region=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_patient_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Set truncation if 'SpinalCord'.
        truncate = True if 'SpinalCord' in localiser[0] else False

        # Make prediction.
        pred = get_patient_localiser_prediction(dataset, pat_id, localiser, truncate=truncate)

    if centre_of is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                centre_of = patient.region_data(region=centre_of)[centre_of]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = patient.region_data(region=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]
    
    # Plot.
    plot_localiser_prediction(pat_id, spacing, pred, centre_of=centre_of, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    localisers: Union[types.ModelName, List[types.ModelName]],
    segmenters: Union[types.ModelName, List[types.ModelName]],
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    load_loc_pred: bool = True,
    load_seg_pred: bool = True,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,
    show_ct: bool = True,
    seg_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    **kwargs) -> None:
    # Convert args to 'list'.
    if type(localisers) == tuple:
        localisers = [localisers]
    if type(segmenters) == tuple:
        segmenters = [segmenters]
    if type(regions) == str:
        regions = [regions]
    assert len(localisers) == len(segmenters)
    n_models = len(localisers)
    model_names = tuple(f'model-{i}' for i in range(n_models))

    # Infer 'pred_regions' from localiser model names.
    pred_regions = [l[0].split('-')[1] for l in localisers]
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * n_models
    else:
        assert len(seg_spacings) == n_models
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(region=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    pred_data = {}
    for i in range(n_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        model_name = model_names[i]
        pred_region = pred_regions[i]
        seg_spacing = seg_spacings[i]

        # Load/make localiser prediction.
        loc_centre = None
        if load_loc_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser, raise_error=False)
            if loc_centre is None:
                logging.info(f"No prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
        if loc_centre is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            truncate = True if pred_region == 'SpinalCord' else False
            create_patient_localiser_prediction(dataset, pat_id, localiser, truncate=truncate)
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)

        # Get segmenter prediction.
        pred = None
        # Attempt load.
        if load_seg_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter, raise_error=False)
            if pred is None:
                logging.info(f"No prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
        # Make prediction if didn't/couldn't load.
        if pred is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            create_patient_segmenter_prediction(dataset, pat_id, pred_region, localiser, segmenter, seg_spacing=seg_spacing)           # Handle multiple spacings.
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)

        loc_centres.append(loc_centre)
        pred_data[model_name] = pred

    if centre_of is not None:
        if centre_of == 'model':
            assert n_models == 1
            centre_of = pred_data[model_names[0]]
        elif type(centre_of) == str:
            match = re.search(MODEL_SELECT_PATTERN, centre_of)
            if match is not None:
                model_i = int(match.group(1))
                assert model_i < n_models
                centre_of = pred_data[model_names[model_i]]
            elif region_data is None or centre_of not in region_data:
                centre_of = patient.region_data(region=centre_of)[centre_of]

    if type(crop) == str:
        if crop == 'model':
            assert n_models == 1
            crop = pred_data[model_names[0]]
        else:
            match = re.search(MODEL_SELECT_PATTERN, crop)
            if match is not None:
                model_i = int(match.group(1))
                assert model_i < n_models
                crop = pred_data[model_names[model_i]]
            elif region_data is None or crop not in region_data:
                crop = patient.region_data(region=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]
    
    # Plot.
    plot_segmenter_prediction(pat_id, spacing, pred_data, centre_of=centre_of, crop=crop, ct_data=ct_data, loc_centres=loc_centres, region_data=region_data, **kwargs)
