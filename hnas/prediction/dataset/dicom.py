import numpy as np
import os
import pydicom as dcm
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from hnas import dataset as ds
from hnas.dataset.dicom import ROIData, RTSTRUCTConverter
from hnas import logging
from hnas.models.systems import Localiser, Segmenter
from hnas.regions import to_255, RegionColours
from hnas import utils
from hnas import types

def get_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.Model,
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)

    # Load the patient data.
    set = ds.get(dataset, 'dicom')
    input = set.patient(pat_id).ct_data()
    input_size = input.shape
    spacing = set.patient(pat_id).ct_spacing()

    # Check patient FOV.
    fov = np.array(input_size) * spacing
    loc_fov = np.array(loc_size) * loc_spacing
    for axis in len(fov):
        if fov[axis] > loc_fov[axis]:
            raise ValueError(f"Patient FOV '{fov}', larger than localiser FOV '{loc_fov}'.")

    # Resample/crop data for network.
    input = resample_3D(input, spacing, loc_spacing)
    pre_crop_size = input.shape

    # Shape the image so it'll fit the network.
    input = centre_crop_or_pad_3D(input, loc_size, fill=input.min())

    # Get localiser result.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = localiser(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Reverse the resample/crop.
    pred = centre_crop_or_pad_3D(pred, pre_crop_size)
    pred = resample_3D(pred, loc_spacing, spacing)
    
    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)
    return pred

def create_localiser_predictions(
    dataset: str,
    localiser: Tuple[str, str, str],
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    region: types.PatientRegions = 'all') -> None:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load patients.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(region=region)

    # Load models.
    localiser_args = localiser
    localiser = Localiser.load(*localiser)

    for pat in tqdm(pats):
        # Make prediction.
        _, data = get_localiser_prediction(set, pat, localiser, loc_size, loc_spacing, device=device, return_seg=True)

        # Save in folder.
        spacing = set.patient(pat).ct_spacing()
        affine = np.ndarray([
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1]
        ])
        img = nib.Nifti1Image(data, affine)
        filepath = os.path.join(set.path, 'predictions', 'localiser', f"{localiser_args[0]}-{segmenter_args[0]}", f"{localiser_args[1]}-{segmenter_args[1]}", f"{localiser_args[2]}-{segmenter_args[2]}", f"{pat}.nii.gz") 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

def create_segmenter_predictions(
    dataset: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    region: types.PatientRegions = 'all') -> None:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load patients.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(region=region)

    # Load models.
    localiser_args = localiser
    segmenter_args = segmenter
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)

    # Create RTSTRUCT info.
    rt_info = {
        'label': 'HNAS',
        'institution-name': 'HNAS'
    }

    for pat in tqdm(pats):
        # Get segmentation.
        seg = get_patient_segmentation(set, pat, localiser, segmenter, device=device)

        # Load reference CT dicoms.
        cts = set.patient(pat).get_cts()

        # Create RTSTRUCT dicom.
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts, rt_info)

        # Create ROI data.
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=seg,
            frame_of_reference_uid=rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name=region
        )

        # Add ROI.
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)

        # Save in folder.
        filename = f"{pat}.dcm"
        filepath = os.path.join(set.path, 'predictions', 'two-stage', f"{localiser_args[0]}-{segmenter_args[0]}", f"{localiser_args[1]}-{segmenter_args[1]}", f"{localiser_args[2]}-{segmenter_args[2]}", filename) 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct.save_as(filepath)

def create_dataset(
    dataset: str,
    device: torch.device = torch.device('cpu'),
    output_dataset: Optional[str] = None,
    use_gpu: bool = True) -> None:
    """
    effect: generates a DICOMDataset of predictions.
    args:
        dataset: the dataset to create predictions from.
    kwargs:
        device: the device to perform inference on.
        output_dataset: the name of the dataset to hold the predictions.
        use_gpu: use GPU for matrix calculations.
    """
    # Load patients.
    source_ds = ds.get(dataset, 'dicom')
    pats = source_ds.list_patients()

    # Re/create pred dataset.
    pred_ds_name = output_dataset if output_dataset else f"{dataset}-pred"
    recreate(pred_ds_name)
    ds_pred = ds.get(pred_ds_name, type_str='dicom')

    # Create RTSTRUCT info.
    rt_info = {
        'label': 'HNAS',
        'institution-name': 'HNAS'
    }

    for pat in tqdm(pats):
        # Get segmentation.
        seg = get_patient_segmentation(source_ds, pat, device=device)

        # Load reference CT dicoms.
        cts = ds.patient(pat).get_cts()

        # Create RTSTRUCT dicom.
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts, rt_info)

        # Create ROI data.
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=seg,
            frame_of_reference_uid=rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name='Parotid_L'
        )

        # Add ROI.
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)

        # Save in new 'pred' dataset.
        filename = f"{pat}.dcm"
        filepath = os.path.join(ds_pred.path, 'raw', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct.save_as(filepath)

def load_segmenter_predictions(
    dataset: str,
    pat_id: str,
    model: str,
    regions: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    if type(regions) == str:
        regions = [regions]

    # Load ref CTs.
    set = ds.get(dataset, 'dicom')
    region_map = set.region_map
    patient = set.patient(pat_id)
    ref_cts = patient.get_cts()

    # Get region info.
    filepath = os.path.join(set.path, 'predictions', model, f'{pat_id}.dcm')
    rtstruct = dcm.read_file(filepath)
    region_names = RTSTRUCTConverter.get_roi_names(rtstruct)
    def to_internal(name):
        if region_map is None:
            return name
        else:
            return region_map.to_internal(name)
    name_map = dict((to_internal(name), name) for name in region_names)

    # Extract data.
    preds = []
    for region in regions:
        pred = RTSTRUCTConverter.get_roi_data(rtstruct, name_map[region], ref_cts)
        preds.append(pred)
    
    # Determine return type.
    if len(preds) == 1:
        return preds[0]
    else:
        return preds
