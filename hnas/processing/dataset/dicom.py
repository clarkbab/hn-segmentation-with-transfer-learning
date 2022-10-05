import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
from typing import Optional
from tqdm import tqdm

from hnas.dataset.dicom import DICOMDataset
from hnas.dataset.nifti import recreate as recreate_nifti
from hnas import logging
from hnas.regions import region_to_list
from hnas.types import PatientRegions
from hnas.utils import append_row, save_csv

from .dataset import write_flag

def convert_to_nifti(
    dataset: str,
    region: PatientRegions = 'all',
    anonymise: bool = False) -> None:
    # Create NIFTI dataset.
    dicom_set = DICOMDataset(dataset)
    nifti_set = recreate_nifti(dataset)
    logging.info(f"Converting dataset '{dataset}' to dataset '{nifti_set}', with region '{region}' and anonymise '{anonymise}'.")

    regions = region_to_list(region)

    # Load all patients.
    pat_ids = dicom_set.list_patients(region=regions)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        df = pd.DataFrame(pat_ids, columns=['patient-id']).reset_index().rename(columns={ 'index': 'anon-id' })

        # Save map.
        save_csv(df, 'anon-maps', f'{dataset.name}.csv', overwrite=True)

    for pat_id in tqdm(pat_ids):
        # Get anonymous ID.
        if anonymise:
            anon_id = df[df['patient-id'] == pat_id].index.values[0]
            filename = f'{anon_id}.nii.gz'
        else:
            filename = f'{pat_id}.nii.gz'

        # Create CT NIFTI.
        patient = dicom_set.patient(pat_id)
        data = patient.ct_data
        spacing = patient.ct_spacing
        offset = patient.ct_offset
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(nifti_set.path, 'data', 'ct', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        pat_regions = patient.list_regions()
        pat_regions = [r for r in pat_regions if r in regions]
        region_data = patient.region_data(region=pat_regions)
        for region, data in region_data.items():
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(nifti_set.path, 'data', 'regions', region, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

    # Indicate success.
    write_flag(nifti_set, '__CONVERT_FROM_NIFTI_END__')

def convert_to_nifti_multiple_studies(
    dataset: str,
    dicom_dataset: Optional[str] = None,
    region: PatientRegions = 'all',
    anonymise: bool = False) -> None:
    regions = region_to_list(region)

    # Create NIFTI dataset.
    nifti_set = recreate_nifti(dataset)
    logging.arg_log('Converting dataset to NIFTI', ('dataset', 'regions', 'anonymise'), (dataset, regions, anonymise))

    # Get all patients.
    dicom_dataset = dataset if dicom_dataset is None else dicom_dataset
    set = DICOMDataset(dicom_dataset)
    filepath = os.path.join(set.path, 'patient-studies.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"File '<dataset>/patient-studies.csv' not found.")
    study_df = pd.read_csv(filepath, dtype={ 'patient-id': str })
    pat_ids = list(sorted(np.unique(study_df['patient-id'])))

    if anonymise:
        cols = {
            'patient-id': str,
            'origin-dataset': str,
            'origin-patient-id': str,
            'origin-study-id': str
        }
        df = pd.DataFrame(columns=cols.keys())

    for i, pat_id in enumerate(tqdm(pat_ids)):
        # Get study IDs.
        study_ids = study_df[study_df['patient-id'] == pat_id]['study-id'].values

        for j, study_id in enumerate(study_ids):
            # Get ID.
            if anonymise:
                nifti_id = f'{i}-{j}'
            else:
                nifti_id = f'{pat_id}-{j}'

            # Add row to anon index.
            if anonymise:
                data = {
                    'patient-id': nifti_id,
                    'origin-dataset': dicom_dataset,
                    'origin-patient-id': pat_id,
                    'origin-study-id': study_id,
                }
                df = append_row(df, data)

            # Create CT NIFTI for study.
            pat = set.patient(pat_id)
            study = pat.study(study_id)
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            ct_offset = study.ct_offset
            affine = np.array([
                [ct_spacing[0], 0, 0, ct_offset[0]],
                [0, ct_spacing[1], 0, ct_offset[1]],
                [0, 0, ct_spacing[2], ct_offset[2]],
                [0, 0, 0, 1]])
            img = Nifti1Image(ct_data, affine)
            filepath = os.path.join(nifti_set.path, 'data', 'ct', f'{nifti_id}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

            # Create region NIFTIs for study.
            pat_regions = study.list_regions()
            pat_regions = [r for r in pat_regions if r in regions]
            region_data = study.region_data(region=pat_regions)
            for region, data in region_data.items():
                img = Nifti1Image(data.astype(np.int32), affine)
                filepath = os.path.join(nifti_set.path, 'data', 'regions', region, f'{nifti_id}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create RTDOSE NIFTIs for study.
            dose_data = study.dose_data
            if dose_data is not None:
                dose_spacing = study.dose_spacing
                dose_offset = study.dose_offset
                affine = np.array([
                    [dose_spacing[0], 0, 0, dose_offset[0]],
                    [0, dose_spacing[1], 0, dose_offset[1]],
                    [0, 0, dose_spacing[2], dose_offset[2]],
                    [0, 0, 0, 1]])
                img = Nifti1Image(dose_data, affine)
                filepath = os.path.join(nifti_set.path, 'data', 'dose', f'{nifti_id}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

    if anonymise:
        filepath = os.path.join(nifti_set.path, 'anon-index.csv') 
        df.to_csv(filepath, index=False)

    # Indicate success.
    write_flag(nifti_set, '__CONVERT_FROM_NIFTI_END__')
