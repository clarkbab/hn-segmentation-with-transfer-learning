import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import pandas as pd
import pydicom as dcm
import os
import shutil
from tqdm import tqdm
from typing import Callable

from hnas import logging
from hnas import types

from ..nifti import recreate as recreate_nifti
from .dicom_dataset import DICOMDataset

def convert_to_nifti(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Load all patients.
    set = DICOMDataset(dataset)
    pats = set.list_patients(region=regions)

    # Create NIFTI dataset.
    nifti_set = recreate_nifti(dataset)

    for pat in tqdm(pats):
        # Create CT NIFTI.
        patient = set.patient(pat)
        data = patient.ct_data()
        spacing = patient.ct_spacing()
        offset = patient.ct_offset()
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(nifti_ds.path, 'ct', f'{pat}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        pat_regions = patient.list_regions(whitelist=regions)
        region_data = patient.region_data(region=pat_regions)
        for region, data in region_data.items():
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(nifti_ds.path, region, f'{pat}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)
