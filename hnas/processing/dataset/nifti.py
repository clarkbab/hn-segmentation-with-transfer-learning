import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import List, Optional, Union

from hnas import config
from hnas import dataset as ds
from hnas.dataset.dicom import DICOMDataset, ROIData, RTSTRUCTConverter
from hnas.dataset.nifti import NIFTIDataset
from hnas.dataset.nifti import recreate as recreate_nifti
from hnas.dataset.training import TrainingDataset, exists
from hnas.dataset.training import create as create_training
from hnas.dataset.training import recreate as recreate_training
from hnas.loaders import Loader
from hnas import logging
from hnas.models import replace_checkpoint_alias
from hnas.prediction.dataset.nifti import create_localiser_prediction, create_segmenter_prediction, load_segmenter_prediction
from hnas.regions import RegionColours, RegionNames, to_255
from hnas.regions import region_to_list
from hnas.reporting.loaders import load_loader_manifest
from hnas.transforms import resample_3D, top_crop_or_pad_3D
from hnas import types
from hnas.utils import append_row, arg_to_list, load_csv, save_csv

def convert_to_training(
    dataset: str,
    region: types.PatientRegions,
    create_data: bool = True,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    output_size: Optional[types.ImageSize3D] = None,
    output_spacing: Optional[types.ImageSpacing3D] = None,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    training_dataset: Optional[str] = None) -> None:
    logging.arg_log('Converting to training', ('dataset', 'region'), (dataset, region))
    regions = region_to_list(region)

    # Create the dataset.
    dest_dataset = dataset if training_dataset is None else training_dataset
    if exists(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate_training(dest_dataset)
        else:
            created = False
            set_t = TrainingDataset(dest_dataset)
            _destroy_flag(set_t, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create_training(dest_dataset)
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_START__')

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'output-size': [str(output_size)] if output_size is not None else ['None'],
            'output-spacing': [str(output_spacing)] if output_spacing is not None else ['None'],
            'regions': [str(regions)],
        })
        params_df.to_csv(filepath, index=False)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'output-size': [str(output_size)] if output_size is not None else ['None'],
                'output-spacing': [str(output_spacing)] if output_spacing is not None else ['None'],
                'regions': [str(regions)],
            })
            params_df.to_csv(filepath, index=False)

    # Load patients.
    set = NIFTIDataset(dataset)
    pat_ids = set.list_patients(region=regions)

    # Get exclusions.
    exc_df = set.excluded_labels

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'group-id': float,
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())
    index = index.astype(cols)

    # Load patient grouping if present.
    group_df = set.group_index

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            # Load input data.
            patient = set.patient(pat_id)
            spacing = patient.ct_spacing
            input = patient.ct_data

            # Resample input.
            if output_spacing:
                input = resample_3D(input, spacing=spacing, output_spacing=output_spacing)

            # Crop/pad.
            if output_size:
                # Log warning if we're cropping the FOV as we're losing information.
                if log_warnings:
                    if output_spacing:
                        fov_spacing = output_spacing
                    else:
                        fov_spacing = spacing
                    fov = np.array(input.shape) * fov_spacing
                    new_fov = np.array(output_size) * fov_spacing
                    for axis in range(len(output_size)):
                        if fov[axis] > new_fov[axis]:
                            logging.warning(f"Patient '{patient}' had FOV '{fov}', larger than new FOV after crop/pad '{new_fov}' for axis '{axis}'.")

                # Perform crop/pad.
                input = top_crop_or_pad_3D(input, output_size, fill=input.min())

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_region(region):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = patient.region_data(region=region)[region]

                # Resample data.
                if output_spacing:
                    label = resample_3D(label, spacing=spacing, output_spacing=output_spacing)

                # Crop/pad.
                if output_size:
                    label = top_crop_or_pad_3D(label, output_size)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    empty = False
                    __create_training_label(set_t, i, region, label)
                else:
                    empty = True

                # Add index entry.
                if group_df is not None:
                    tdf = group_df[group_df['patient-id'] == pat_id]
                    if len(tdf) == 0:
                        group_id = np.nan
                    else:
                        assert len(tdf) == 1
                        group_id = tdf.iloc[0]['group-id']
                else:
                    group_id = np.nan
                data = {
                    'dataset': set_t.name,
                    'sample-id': i,
                    'group-id': group_id,
                    'origin-dataset': set.name,
                    'origin-patient-id': pat_id,
                    'region': region,
                    'empty': empty
                }
                index = append_row(index, data)

    end = time()

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Indicate success.
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    _print_time(set_t, hours)

def create_excluded_brainstem(
    dataset: str,
    dest_dataset: str) -> None:
    # Copy dataset to destination.
    set = NIFTIDataset(dataset)
    dest_set = recreate_nifti(dest_dataset)
    os.rmdir(dest_set.path)
    shutil.copytree(set.path, dest_set.path)

    cols = {
        'patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Get patient with 'Brain' label.
    pat_ids = dest_set.list_patients(region='Brain')
    for pat_id in tqdm(pat_ids):
        # Skip if no 'Brainstem'.
        pat = dest_set.patient(pat_id)
        if not pat.has_region('Brainstem'):
            continue

        # Load label data.
        data = pat.region_data(region=['Brain', 'Brainstem'])

        # Perform exclusion.
        brain_data = data['Brain'] & ~data['Brainstem']

        # Write new label.
        ct_spacing = pat.ct_spacing
        ct_offset = pat.ct_offset
        affine = np.array([
            [ct_spacing[0], 0, 0, ct_offset[0]],
            [0, ct_spacing[1], 0, ct_offset[1]],
            [0, 0, ct_spacing[2], ct_offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(brain_data.astype(np.int32), affine)
        filepath = os.path.join(dest_set.path, 'data', 'regions', 'Brain', f'{pat_id}.nii.gz')
        nib.save(img, filepath)

        # Add to index.
        data = {
            'patient-id': pat_id
        }
        df = append_row(df, data)

    # Save index.
    filepath = os.path.join(dest_set.path, 'excl-index.csv')
    df.to_csv(filepath, index=False)

def _destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def _write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def _print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NIFTI_TIME_HOURS_{hours}__')
    Path(path).touch()

def __create_training_input(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(dataset.path, 'data', 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def __create_training_label(
    dataset: 'Dataset',
    index: int,
    region: str,
    data: np.ndarray) -> None:
    # Save the label data.
    filepath = os.path.join(dataset.path, 'data', 'labels', region, f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def convert_segmenter_predictions_to_dicom_from_all_patients(
    n_pats: int,
    anonymise: bool = True) -> None:
    logging.arg_log('Converting segmenter predictions to DICOM', ('n_pats', 'anonymise'), (n_pats, anonymise))

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')
    df = df.astype({ 'patient-id': str })
    df = df.head(n_pats)

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI-HN',
        'institution-name': 'PMCC-AI-HN'
    }

    # Create index.
    if anonymise:
        cols = {
            'patient-id': str,
            'anon-id': str
        }
        index_df = pd.DataFrame(columns=cols.keys())

    for i, (dataset, pat_id) in tqdm(df.iterrows()):
        # Get ROI ID from DICOM dataset.
        nifti_set = NIFTIDataset(dataset)
        pat_id_dicom = nifti_set.patient(pat_id).patient_id
        set_dicom = DICOMDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.get_rtstruct()
        info_gt = RTSTRUCTConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RTSTRUCTConverter.create_rtstruct(cts, default_rt_info)
        frame_of_reference_uid = rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

        for region in RegionNames:
            # Load prediction.
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nifti', dataset, pat_id, f'{region}.npz')
            pred = np.load(filepath)['data']
            
            # Match ROI number to ground truth, otherwise assign next available integer.
            if region not in region_map_gt:
                for j in range(1, 1000):
                    if j not in region_map_gt.values():
                        region_map_gt[region] = j
                        break
                    elif j == 999:
                        raise ValueError(f'Unlikely')
            roi_number = region_map_gt[region]

            # Add ROI data.
            roi_data = ROIData(
                colour=list(to_255(getattr(RegionColours, region))),
                data=pred,
                frame_of_reference_uid=frame_of_reference_uid,
                name=region,
                number=roi_number
            )
            RTSTRUCTConverter.add_roi(rtstruct_pred, roi_data, cts)

        # Add index row.
        if anonymise:
            anon_id = f'PMCC_AI_HN_{i + 1:03}'
            data = {
                'patient-id': pat_id,
                'anon-id': anon_id
            }
            index_df = append_row(index_df, data)

        # Save pred RTSTRUCT.
        pat_id_folder = anon_id if anonymise else pat_id_dicom
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'rtstruct', 'pred.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if anonymise:
            rtstruct_pred.PatientID = anon_id
            rtstruct_pred.PatientName = anon_id
        rtstruct_pred.save_as(filepath)

        # Copy CTs.
        for j, path in enumerate(patient_dicom.default_rtstruct.ref_ct.paths):
            ct = dcm.read_file(path)
            if anonymise:
                ct.PatientID = anon_id
                ct.PatientName = anon_id
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'ct', f'{j}.dcm')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            ct.save_as(filepath)

        # Copy ground truth RTSTRUCT.
        rtstruct_gt = patient_dicom.default_rtstruct.get_rtstruct()
        if anonymise:
            rtstruct_gt.PatientID = anon_id
            rtstruct_gt.PatientName = anon_id
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'rtstruct', 'gt.dcm')
        rtstruct_gt.save_as(filepath)
    
    # Save index.
    if anonymise:
        save_csv(index_df, 'transfer-learning', 'data', 'predictions', 'dicom', 'index.csv')

def convert_segmenter_predictions_to_dicom_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_manifest: bool = False,
    use_model_manifest: bool = False) -> None:
    # Get unique name.
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)
    logging.info(f"Converting segmenter predictions to DICOM for '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Build test loader.
    if use_loader_manifest:
        man_df = load_loader_manifest(datasets, region, n_folds=n_folds, test_fold=test_fold)
        samples = man_df[['dataset', 'patient-id']].to_numpy()
    else:
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
        test_dataset = test_loader.dataset
        samples = [test_dataset.__get_item(i) for i in range(len(test_dataset))]

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI-HN',
        'institution-name': 'PMCC-AI-HN'
    }

    # Create prediction RTSTRUCTs.
    for dataset, pat_id_nifti in tqdm(samples):
        # Get ROI ID from DICOM dataset.
        nifti_set = NIFTIDataset(dataset)
        pat_id_dicom = nifti_set.patient(pat_id_nifti).patient_id
        set_dicom = DICOMDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.get_rtstruct()
        info_gt = RTSTRUCTConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RTSTRUCTConverter.create_rtstruct(cts, default_rt_info)

        # Load prediction.
        pred = load_patient_segmenter_prediction(dataset, pat_id_nifti, localiser, segmenter)
        
        # Add ROI.
        roi_data = ROIData(
            colour=list(to_255(getattr(RegionColours, region))),
            data=pred,
            frame_of_reference_uid=rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name=region,
            number=region_map_gt[region]        # Patient should always have region (right?) - we created the loaders based on patient regions.
        )
        RTSTRUCTConverter.add_roi(rtstruct_pred, roi_data, cts)

        # Save prediction.
        # Get localiser checkpoint and raise error if multiple.
        # Hack - clean up when/if path limits are removed.
        if config.environ('PETER_MAC_HACK') == 'True':
            base_path = 'S:\\ImageStore\\HN_AI_Contourer\\short\\dicom'
            if dataset == 'PMCC-HN-TEST':
                pred_path = os.path.join(base_path, 'test')
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = os.path.join(base_path, 'train')
        else:
            pred_path = os.path.join(nifti_set.path, 'predictions', 'segmenter')
        filepath = os.path.join(pred_path, *localiser, *segmenter, f'{pat_id_dicom}.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct_pred.save_as(filepath)

def combine_segmenter_predictions_from_all_patients(
    dataset: Union[str, List[str]],
    n_pats: int,
    model_type: str = 'clinical') -> None:
    datasets = arg_to_list(dataset, str)
    logging.arg_log("Combining (NIFTI) segmenter predictions from 'all-patients.csv'", ('dataset', 'n_pats', 'model_type'), (datasets, n_pats, model_type))

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')
    df = df.astype({ 'patient-id': str })
    df = df.head(n_pats)

    cols = {
        'region': str,
        'model': str
    }

    for _, (dataset, pat_id) in tqdm(df.iterrows()):
        index_df = pd.DataFrame(columns=cols.keys())

        for region in RegionNames:
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')

            # Find fold that didn't use this patient for training.
            for test_fold in range(5):
                man_df = load_loader_manifest(datasets, region, test_fold=test_fold)
                man_df = man_df[(man_df.loader == 'test') & (man_df['origin-dataset'] == dataset) & (man_df['origin-patient-id'] == pat_id)]
                if len(man_df) == 1:
                    break
            
            # Select segmenter that didn't use this patient for training.
            if len(man_df) == 1:
                # Patient was excluded when training model for 'test_fold'.
                segmenter = (f'segmenter-{region}-v2', f'{model_type}-fold-{test_fold}-samples-None', 'best')
            elif len(man_df) == 0:
                # This patient region wasn't used for training any models, let's just use the model of the first fold.
                segmenter = (f'segmenter-{region}-v2', f'{model_type}-fold-0-samples-None', 'best') 
            else:
                raise ValueError(f"Found multiple matches in loader manifest for test fold '{test_fold}', dataset '{dataset}', patient '{pat_id}' and region '{region}'.")

            # Add index row.
            data = {
                'region': region,
                'model': f'{model_type}-fold-{test_fold}-samples-None'
            }
            index_df = append_row(index_df, data)

            # Load/create segmenter prediction.
            try:
                pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)
            except ValueError as e:
                logging.info(str(e))
                create_localiser_prediction(dataset, pat_id, localiser)
                create_segmenter_prediction(dataset, pat_id, localiser, segmenter)
                pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)

            # Copy prediction to new location.
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nifti', dataset, pat_id, f'{region}.npz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=pred)

        # Save patient index.
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nifti', dataset, pat_id, 'index.csv')
        index_df.to_csv(filepath, index=False)
