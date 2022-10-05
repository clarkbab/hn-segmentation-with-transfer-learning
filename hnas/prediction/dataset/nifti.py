from hnas.dataset.nifti.nifti_dataset import NIFTIDataset
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Literal, Optional, Union

from ..prediction import get_localiser_prediction as get_localiser_prediction_base
from hnas import config
from hnas import dataset as ds
from hnas.geometry import get_box, get_extent, get_extent_centre, get_extent_width_mm
from hnas.loaders import Loader
from hnas import logging
from hnas.models import replace_checkpoint_alias
from hnas.models.systems import Localiser, Segmenter
from hnas.regions import RegionNames, get_region_patch_size, truncate_spine
from hnas.reporting.loaders import load_loader_manifest
from hnas.transforms import crop_foreground_3D, crop_or_pad_3D, resample_3D
from hnas import types
from hnas.utils import Timer, append_row, arg_broadcast, arg_to_list, encode, load_csv

def get_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction_base(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def create_localiser_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[Union[int, str], List[Union[int, str]]],
    localiser: Union[types.ModelName, types.Model],
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, [int, str], out_type=str)
    datasets = arg_broadcast(datasets, pat_ids)
    assert len(datasets) == len(pat_ids)

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = ds.get(dataset, 'nifti')
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser.name}'.")

        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser.name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_localiser_predictions_for_first_n_pats(
    n_pats: int,
    region: str,
    localiser: types.ModelName,
    savepath: Optional[str] = None) -> None:
    localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets for region '{region}', first '{n_pats}' patients in 'all-patients.csv'.")

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Get dataset/patient IDs.
    create_localiser_prediction(*df, localiser, device=device, savepath=savepath)

def create_localiser_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Timing table data.
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'device': device.type
            }

            with timer.record(timing, data):
                create_localiser_prediction(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def load_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    exists_only: bool = False) -> Union[np.ndarray, bool]:
    localiser = replace_checkpoint_alias(*localiser)

    # Load prediction.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, str(pat_id), *localiser, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")

    pred = np.load(filepath)['data']
    return pred

def load_localiser_predictions_timings(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    localiser = replace_checkpoint_alias(*localiser)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser, f'timing-folds-{n_folds}-test-{test_fold}-device-{device}.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction timings not found for datasets '{datasets}', region '{region}', and localiser '{localiser}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def load_localiser_centre(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName) -> types.Point3D:
    spacing = NIFTIDataset(dataset).patient(pat_id).ct_spacing

    # Get localiser prediction.
    pred = load_localiser_prediction(dataset, pat_id, localiser)

    # Apply cropping for SpinalCord predictions that are "too long" on caudal end.
    # Otherwise the segmentation patch won't cover the top of the SpinalCord.
    region = localiser[0].split('-')[1]         # Infer region.
    if region == 'SpinalCord':
        pred = truncate_spine(pred, spacing)

    # Get localiser pred centre.
    ext_centre = get_extent_centre(pred)

    return ext_centre

def get_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    loc_centre: types.Point3D,
    segmenter: Union[types.Model, types.ModelName],
    probs: bool = False,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    # Load model.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)

    # Load patient CT data and spacing.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample_3D(input, spacing=spacing, output_spacing=seg_spacing) 

    # Get localiser centre on downsampled image.
    scaling = np.array(spacing) / seg_spacing
    loc_centre = tuple(int(el) for el in scaling * loc_centre)

    # Extract segmentation patch.
    resampled_size = input.shape
    region = segmenter.name[0].split('-')[1]        # Infer region from model name.
    patch_size = get_region_patch_size(region, seg_spacing)
    patch = get_box(loc_centre, patch_size)
    input = crop_or_pad_3D(input, patch, fill=input.min())

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = segmenter(input, probs=probs)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Crop/pad to the resampled size, i.e. before patch extraction.
    rev_patch_min, rev_patch_max = patch
    rev_patch_min = tuple(-np.array(rev_patch_min))
    rev_patch_max = tuple(np.array(rev_patch_min) + resampled_size)
    rev_patch_box = (rev_patch_min, rev_patch_max)
    pred = crop_or_pad_3D(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample_3D(pred, spacing=seg_spacing, output_spacing=spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def create_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    localiser: types.ModelName,
    segmenter: Union[types.Model, types.ModelName],
    device: Optional[torch.device] = None,
    probs: bool = False,
    raise_error: bool = False,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, str)
    datasets = arg_broadcast(dataset, pat_ids, arg_type=str)
    localiser = replace_checkpoint_alias(*localiser)
    assert len(datasets) == len(pat_ids)

    # Load segmenter.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = ds.get(dataset, 'nifti')
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser}', segmenter '{segmenter.name}'.")

        # Load localiser centre.
        loc_centre = load_localiser_centre(dataset, pat_id, localiser)

        # Get segmenter prediction.
        if loc_centre is None:
            # Create empty pred.
            if raise_error:
                raise ValueError(f"No 'loc_centre' returned from localiser.")
            else:
                ct_data = set.patient(pat_id).ct_data
                pred = np.zeros_like(ct_data, dtype=bool) 
        else:
            pred = get_segmenter_prediction(dataset, pat_id, loc_centre, segmenter, device=device)

        # Save segmentation.
        if probs:
            filename = 'pred-prob.npz'
        else:
            filename = 'pred.npz'
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, pat_id, *localiser, *segmenter.name, filename)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_segmenter_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Timing table data.
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'device': device.type
            }

            with timer.record(timing, data):
                create_segmenter_prediction(dataset, pat_id, localiser, segmenter, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def load_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    exists_only: bool = False,
    use_model_manifest: bool = False) -> Union[np.ndarray, bool]:
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)

    # Load segmentation.
    filepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, str(pat_id), *localiser, *segmenter, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'. Path: {filepath}")

    pred = np.load(filepath)['data']
    return pred

def load_segmenter_predictions_timings(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = replace_checkpoint_alias(*segmenter)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser, *segmenter, f'timing-folds-{n_folds}-test-{test_fold}-device-{device}.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction timings not found for datasets '{datasets}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def save_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    data: np.ndarray) -> None:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    np.savez_compressed(filepath, data=data)

def create_two_stage_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[Union[int, List[int], Literal['all']]] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    if test_fold == 'all':
        test_folds = list(range(n_folds))
    elif type(test_fold) == int:
        test_folds = [test_fold]
    else:
        test_folds = test_fold
    logging.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        loc_timer = Timer(cols)
        seg_timer = Timer(cols)

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

        # Make predictions.
        for pat_desc_b in tqdm(iter(test_loader)):
            if type(pat_desc_b) == torch.Tensor:
                pat_desc_b = pat_desc_b.tolist()
            for pat_desc in pat_desc_b:
                dataset, pat_id = pat_desc.split(':')

                # Timing table data.
                data = {
                    'fold': test_fold,
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'region': region,
                    'device': device.type
                }

                with loc_timer.record(timing, data):
                    create_localiser_prediction(dataset, pat_id, localiser, device=device)

                with seg_timer.record(timing, data):
                    create_segmenter_prediction(dataset, pat_id, localiser.name, segmenter, device=device)

        # Save timing data.
        if timing:
            filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            loc_timer.save(filepath)
            filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser.name, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            seg_timer.save(filepath)
