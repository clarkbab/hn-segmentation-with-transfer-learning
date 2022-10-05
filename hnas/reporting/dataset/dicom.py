import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List

from hnas import dataset as ds
from hnas.evaluation.dataset.dicom import evaluate_model
from hnas.geometry import get_extent
from hnas import types
from hnas.utils import append_row, encode

def create_evaluation_report(
    name: str,
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    region: str) -> None:
    # Save report.
    eval_df = evaluate_model(dataset, localiser, segmenter, region)
    set = ds.get(dataset, 'dicom')
    filename = f"{name}.csv"
    filepath = os.path.join(set.path, 'reports', 'evaluation', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    eval_df.to_csv(filepath)

def get_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # Get patients.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(region=regions)

    cols = {
        'patient-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Load values.
        patient = set.patient(pat)
        size = patient.ct_size()
        spacing = patient.ct_spacing()

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'patient-id': pat,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get summary.
    df = get_ct_summary(dataset, region=regions)

    # Save summary.
    set = ds.get(dataset, 'dicom')
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    set = ds.get(dataset, 'dicom')
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    return pd.read_csv(filepath)

def region_count(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # List regions.
    set = ds.get(dataset, 'dicom')
    regions_df = set.list_regions(clear_cache=clear_cache)

    # Filter on requested regions.
    def filter_fn(row):
        if type(regions) == str:
            if regions == 'all':
                return True
            else:
                return row['region'] == regions
        else:
            for region in regions:
                if row['region'] == region:
                    return True
            return False
    regions_df = regions_df[regions_df.apply(filter_fn, axis=1)]

    # Generate counts report.
    count_df = regions_df.groupby('region').count().rename(columns={'patient-id': 'count'})
    return count_df

def create_region_count_report(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> None:
    # Generate counts report.
    set = ds.get(dataset, type_str='dicom')
    count_df = region_count(dataset, clear_cache=clear_cache, region=regions)
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath)

def region_overlap(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> int:
    # List regions.
    set = ds.get(dataset, type_str='dicom')
    regions_df = set.list_regions(clear_cache=clear_cache) 
    regions_df = regions_df.drop_duplicates()
    regions_df['count'] = 1
    regions_df = regions_df.pivot(index='patient-id', columns='region', values='count')

    # Filter on requested regions.
    def filter_fn(row):
        if type(regions) == str:
            if regions == 'all':
                return True
            else:
                return row[regions] == 1
        else:
            keep = True
            for region in regions:
                if row[region] != 1:
                    keep = False
            return keep
    regions_df = regions_df[regions_df.apply(filter_fn, axis=1)]
    return len(regions_df) 

def region_summary(
    dataset: str,
    regions: List[str]) -> pd.DataFrame:
    """
    returns: stats on region shapes.
    """
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(region=regions)

    cols = {
        'patient': str,
        'region': str,
        'axis': str,
        'extent-mm': float,
        'spacing-mm': float
    }
    df = pd.DataFrame(columns=cols.keys())

    axes = [0, 1, 2]

    # Initialise empty data structure.
    data = {}
    for region in regions:
        data[region] = {}
        for axis in axes:
            data[region][axis] = []

    for pat in tqdm(pats):
        # Get spacing.
        spacing = set.patient(pat).ct_spacing()

        # Get region data.
        pat_regions = set.patient(pat).list_regions(whitelist=regions)
        rs_data = set.patient(pat).region_data(region=pat_regions)

        # Add extents for all regions.
        for r in rs_data.keys():
            r_data = rs_data[r]
            min, max = get_extent(r_data)
            for axis in axes:
                extent_vox = max[axis] - min[axis]
                extent_mm = extent_vox * spacing[axis]
                data = {
                    'patient': pat,
                    'region': r,
                    'axis': axis,
                    'extent-mm': extent_mm,
                    'spacing-mm': spacing[axis]
                }
                df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_summary_report(
    dataset: str,
    regions: List[str]) -> None:
    # Generate counts report.
    df = region_summary(dataset, regions)

    # Save report.
    filename = 'region-summary.csv'
    set = ds.get(dataset, 'dicom')
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
