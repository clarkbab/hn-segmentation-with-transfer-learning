from fpdf import FPDF, TitleStyle
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage.measurements import label as label_objects
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid1

from hnas import config
from hnas import dataset as ds
from hnas.evaluation.dataset.nifti import load_localiser_evaluation, load_segmenter_evaluation
from hnas.geometry import get_extent, get_extent_centre, get_extent_width_mm
from hnas.loaders import Loader
from hnas import logging
from hnas.models.systems import Localiser
from hnas.plotting.dataset.nifti import plot_patient_localiser_prediction, plot_patient_regions, plot_patient_segmenter_prediction
from hnas.postprocessing import get_largest_cc, get_object, one_hot_encode
from hnas import types
from hnas.utils import append_row, encode

def get_region_summary(
    dataset: str,
    region: str) -> pd.DataFrame:
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=region)

    cols = {
        'dataset': str,
        'patient-id': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        spacing = set.patient(pat).ct_spacing
        label = set.patient(pat).region_data(region=region)[region]

        data = {
            'dataset': dataset,
            'patient-id': pat,
        }

        # Add 'connected' metrics.
        lcc_label = get_largest_cc(label)
        data['metric'] = 'connected'
        data['value'] = 1 if lcc_label.sum() == label.sum() else 0
        df = append_row(df, data)
        data['metric'] = 'connected-largest-p'
        data['value'] = lcc_label.sum() / label.sum()
        df = append_row(df, data)

        # Add OAR extent.
        ext_width_mm = get_extent_width_mm(label, spacing)
        if ext_width_mm is None:
            ext_width_mm = (0, 0, 0)
        data['metric'] = 'extent-mm-x'
        data['value'] = ext_width_mm[0]
        df = append_row(df, data)
        data['metric'] = 'extent-mm-y'
        data['value'] = ext_width_mm[1]
        df = append_row(df, data)
        data['metric'] = 'extent-mm-z'
        data['value'] = ext_width_mm[2]
        df = append_row(df, data)

        # Add extent of largest connected component.
        extent = get_extent(lcc_label)
        if extent:
            min, max = extent
            extent_vox = np.array(max) - min
            extent_mm = extent_vox * spacing
        else:
            extent_mm = (0, 0, 0)
        data['metric'] = 'connected-extent-mm-x'
        data['value'] = extent_mm[0]
        df = append_row(df, data)
        data['metric'] = 'connected-extent-mm-y'
        data['value'] = extent_mm[1]
        df = append_row(df, data)
        data['metric'] = 'connected-extent-mm-z'
        data['value'] = extent_mm[2]
        df = append_row(df, data)

        # Add volume.
        vox_volume = reduce(lambda x, y: x * y, spacing)
        data['metric'] = 'volume-mm3'
        data['value'] = vox_volume * label.sum() 
        df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_summary(
    dataset: str,
    region: str) -> None:
    # List patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=region)

    # Allows us to pass all regions from Spartan 'array' job.
    if len(pats) == 0:
        logging.error(f"No patients with region '{region}' found for dataset '{set}'.")
        return

    logging.info(f"Creating region summary for dataset '{dataset}', region '{region}'.")

    # Generate counts report.
    df = get_region_summary(dataset, region)

    # Save report.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def _get_outlier_cols_func(
    lim_df: pd.DataFrame) -> Callable[[pd.Series], Dict]:
    # Create function to operate on row of 'region summary' table.
    def _outlier_cols(row: pd.Series) -> Dict:
        data = {}

        # Add outlier info.
        for column in lim_df.index:
            col_stats = lim_df.loc[column]
            if row[column] < col_stats['low']:
                outlier = True
                outlier_dir = 'low'
                if col_stats['iqr'] == 0:
                    outlier_n_iqr = np.inf
                else:
                    outlier_n_iqr = (col_stats['q1'] - row[column]) / col_stats['iqr']
            elif row[column] > col_stats['high']:
                outlier = True
                outlier_dir = 'high'
                if col_stats['iqr'] == 0:
                    outlier_n_iqr = np.inf
                else:
                    outlier_n_iqr = (row[column] - col_stats['q3']) / col_stats['iqr']
            else:
                outlier = False
                outlier_dir = ''
                outlier_n_iqr = np.nan

            data[f'{column}-out'] = outlier
            data[f'{column}-out-dir'] = outlier_dir
            data[f'{column}-out-num-iqr'] = outlier_n_iqr

        return data
    return _outlier_cols

def add_region_summary_outliers(
    df: pd.DataFrame,
    columns: List[str]) -> pd.DataFrame:

    # Get outlier limits.
    q1 = df.quantile(0.25)[columns].rename('q1')
    q3 = df.quantile(0.75)[columns].rename('q3')
    iqr = (q3 - q1).rename('iqr')
    low = (q1 - 1.5 * iqr).rename('low')
    high = (q3 + 1.5 * iqr).rename('high')
    lim_df = pd.concat([q1, q3, iqr, low, high], axis=1)

    # Add columns.
    func = _get_outlier_cols_func(lim_df)
    out_df = df.apply(func, axis=1, result_type='expand')
    df = pd.concat([df, out_df], axis=1)
    return df

def load_region_summary(
    dataset: str,
    regions: Optional[Union[str, List[str]]] = None,
    blacklist: bool = False) -> None:
    # Convert to array.
    set = ds.get(dataset, 'nifti')
    if regions is None:
        filepath = os.path.join(set.path, 'reports', 'region-summaries')
        regions = [f.replace('.csv', '') for f in os.listdir(filepath)]
    elif type(regions) == str:
        regions = [regions]

    # Load summary.
    dfs = []
    for region in regions:
        filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
        df = pd.read_csv(filepath)
        df.insert(1, 'region', region)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    # Filter blacklisted records.
    if blacklist:
        filepath = os.path.join(set.path, 'region-blacklist.csv')
        black_df = pd.read_csv(filepath)
        df = df.merge(black_df, how='left', on=['patient-id', 'region'], indicator=True)
        df = df[df['_merge'] == 'left_only']
        df = df.drop(columns='_merge')

    return df

def load_region_count(datasets: Union[str, List[str]]) -> pd.DataFrame:
    if type(datasets) == str:
        datasets = [datasets]

    # Load/concat region counts.
    dfs = []
    for dataset in datasets:
        df = load_region_summary(dataset)
        df = df.groupby('region').count()[['patient-id']].rename(columns={ 'patient-id': 'count' }).reset_index()
        df.insert(0, 'dataset', dataset)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    # Pivot table.
    df = df.pivot(index='dataset', columns='region', values='count').fillna(0).astype(int)
    return df

def get_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    logging.info(f"Creating CT summary for dataset '{dataset}'.")

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=regions)

    cols = {
        'dataset': str,
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
        size = patient.ct_size
        spacing = patient.ct_spacing

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'dataset': dataset,
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
    df = get_ct_summary(dataset)

    # Save summary.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> Optional[pd.DataFrame]:
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return None

def create_prediction_figures(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    logging.info(f"Creating prediction figures for datasets '{datasets}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'.")

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 100
    img_height = 100

    # Create PDF.
    pdf = FPDF()
    pdf.set_section_title_styles(
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=24,
            color=0,
            t_margin=3,
            l_margin=12,
            b_margin=0
        ),
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=18,
            color=0,
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Make predictions.
    for dataset_b, pat_id_b in tqdm(iter(test_loader)):
        if type(pat_id_b) == torch.Tensor:
            pat_id_b = pat_id_b.tolist()
        for dataset, pat_id in zip(dataset_b, pat_id_b):
            # Add patient.
            pdf.add_page()
            pdf.start_section(str(pat_id))

            # Create images.
            views = ['axial', 'coronal', 'sagittal']
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Add image to report.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plot_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter, centre_of='pred', crop='pred', savepath=filepath, show=False, view=view)
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)
                os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'prediction-figures', region, *localiser, *segmenter, f'figures-fold-{test_fold}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_region_figures(
    dataset: str,
    region: str,
    subregions: bool = False) -> None:
    logging.info(f"Creating region figures for dataset '{dataset}', region '{region}'.")

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=region)

    # Keep regions with patients.
    df = load_region_summary(dataset, region=region)
    df = df.pivot(index=['dataset', 'region', 'patient-id'], columns='metric', values='value').reset_index()

    # Add 'extent-mm' outlier info.
    columns = ['extent-mm-x', 'extent-mm-y', 'extent-mm-z']
    df = add_region_summary_outliers(df, columns)

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 100
    img_height = 100

    # Create PDF.
    pdf = FPDF()
    pdf.set_section_title_styles(
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=24,
            color=0,
            t_margin=3,
            l_margin=12,
            b_margin=0
        ),
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=18,
            color=0,
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    for pat in tqdm(pats, leave=False):
        # Skip if patient doesn't have region.
        patient = set.patient(pat)
        if not patient.has_region(region):
            continue

        # Add patient.
        pdf.add_page()
        pdf.start_section(pat)

        # Add region info.
        pdf.start_section('Region Info', level=1)

        # Create table.
        table_t_margin = 50
        table_l_margin = 12
        table_line_height = 2 * pdf.font_size
        table_col_widths = (15, 35, 30, 45, 45)
        pat_info = df[df['patient-id'].astype(str) == pat].iloc[0]
        table_data = [('Axis', 'Extent [mm]', 'Outlier', 'Outlier Direction', 'Outlier Num. IQR')]
        for axis in ['x', 'y', 'z']:
            colnames = {
                'extent': f'extent-mm-{axis}',
                'extent-out': f'extent-mm-{axis}-out',
                'extent-out-dir': f'extent-mm-{axis}-out-dir',
                'extent-out-num-iqr': f'extent-mm-{axis}-out-num-iqr'
            }
            n_iqr = pat_info[colnames['extent-out-num-iqr']]
            format = '.2f' if n_iqr and n_iqr != np.nan else ''
            table_data.append((
                axis,
                f"{pat_info[colnames['extent']]:.2f}",
                str(pat_info[colnames['extent-out']]),
                pat_info[colnames['extent-out-dir']],
                f"{pat_info[colnames['extent-out-num-iqr']]:{format}}",
            ))

        for i, row in enumerate(table_data):
            if i == 0:
                pdf.set_font('Helvetica', 'B', 12)
            else:
                pdf.set_font('Helvetica', '', 12)
            pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
            for j, value in enumerate(row):
                pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add subregion info.
        if subregions:
            # Get object info.
            obj_df = get_object_summary(dataset, pat, region)

            if len(obj_df) > 1:
                pdf.start_section('Subregion Info', level=1)

                # Create table.
                table_t_margin = 105
                table_l_margin = 12
                table_line_height = 2 * pdf.font_size
                table_col_widths = (15, 35, 30, 45, 45)
                table_data = [('ID', 'Volume [mm^3]', 'Volume [prop.]', 'Extent Centre [vox]', 'Extent Width [mm]')]
                for i, row in obj_df.iterrows():
                    table_data.append((
                        str(i),
                        f"{row['volume-mm3']:.2f}",
                        f"{row['volume-p-total']:.2f}",
                        row['extent-centre-vox'],
                        str(tuple([round(e, 2) for e in eval(row['extent-mm'])]))
                    ))
                for i, row in enumerate(table_data):
                    if i == 0:
                        pdf.set_font('Helvetica', 'B', 12)
                    else:
                        pdf.set_font('Helvetica', '', 12)
                    pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
                    for j, value in enumerate(row):
                        pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add region images.
        pdf.add_page()
        pdf.start_section(f'Region Images', level=1)

        # Create images.
        views = ['axial', 'coronal', 'sagittal']
        img_coords = (
            (img_l_margin, img_t_margin),
            (img_l_margin + img_width, img_t_margin),
            (img_l_margin, img_t_margin + img_height)
        )
        for view, page_coord in zip(views, img_coords):
            # Set figure.
            savepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plot_patient_regions(dataset, pat, centre_of=region, colours=['y'], crop=region, region=region, show_extent=True, savepath=savepath, view=view)

            # Add image to report.
            pdf.image(savepath, *page_coord, w=img_width, h=img_height)
            os.remove(savepath)

        # Add subregion images.
        if subregions and len(obj_df) > 1:
            for i, row in obj_df.iterrows():
                pdf.add_page()
                pdf.start_section(f'Subregion {i} Images', level=1)

                # Create images.
                views = ['axial', 'coronal', 'sagittal']
                img_coords = (
                    (img_l_margin, img_t_margin),
                    (img_l_margin + img_width, img_t_margin),
                    (img_l_margin, img_t_margin + img_height)
                )
                for view, page_coord in zip(views, img_coords):
                    # Set figure.
                    def postproc(a: np.ndarray):
                        return get_object(a, i)
                    plot_patient_regions(dataset, pat, centre_of=region, colours=['y'], postproc=postproc, region=region, show_extent=True, view=view, window=(3000, 500))

                    # Save temp file.
                    filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                    plt.savefig(filepath)
                    plt.close()

                    # Add image to report.
                    pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                    # Delete temp file.
                    os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'region-figures', f'{region}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def get_object_summary(
    dataset: str,
    patient: str,
    region: str) -> pd.DataFrame:
    # Get objects.
    pat = ds.get(dataset, 'nifti').patient(patient)
    spacing = pat.ct_spacing
    label = pat.region_data(region=region)[region]
    objs, n_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = one_hot_encode(objs)
    
    cols = {
        'extent-centre-vox': str,
        'extent-mm': str,
        'extent-vox': str,
        'volume-mm3': float,
        'volume-p-total': float,
        'volume-vox': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    tot_voxels = label.sum()
    for i in range(n_objs):
        obj = objs[:, :, :, i]
        data = {}

        # Get extent.
        min, max = get_extent(obj)
        width = tuple(np.array(max) - min)
        width_mm = tuple(np.array(spacing) * width)
        data['extent-mm'] = str(width_mm)
        data['extent-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = get_extent_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        n_voxels = obj.sum()
        volume = n_voxels * vox_volume
        data['volume-vox'] = n_voxels
        data['volume-p-total'] = n_voxels / tot_voxels
        data['volume-mm3'] = volume
        df = append_row(df, data)

    df = df.astype(cols)
    return df

def create_localiser_figures(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str]) -> None:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    logging.info(f"Creating localiser figures for dataset '{dataset}', region '{region}' and localiser '{localiser}'.")

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=region)

    # Exit if region not present.
    set_regions = list(sorted(set.list_regions().region.unique()))
    if region not in set_regions:
        logging.info(f"No region '{region}' present in dataset '{dataset}'.")

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    # Create PDF.
    pdf = FPDF()
    pdf.set_section_title_styles(
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=24,
            color=0,
            t_margin=3,
            l_margin=12,
            b_margin=0
        ),
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=18,
            color=0,
            t_margin=12,
            l_margin=12,
            b_margin=0
        )
    ) 
    
    # Get errors for the region based upon 'extent-dist-x/y/z' metrics.
    eval_df = load_localiser_evaluation(dataset, localiser)
    error_df = eval_df[eval_df.metric.str.contains('extent-dist-')]
    error_df = error_df[(error_df.value.isnull()) | (error_df.value > 0)]

    # Add errors section.
    pdf.add_page()
    pdf.start_section('Errors')

    # Add table.
    table_t_margin = 45
    table_l_margin = 12
    table_line_height = 2 * pdf.font_size
    table_col_widths = (40, 40, 40)
    table_data = [('Patient', 'Metric', 'Value')]
    for _, row in error_df.iterrows():
        table_data.append((
            row['patient-id'],
            row.metric,
            f'{row.value:.3f}'
        ))
    for i, row in enumerate(table_data):
        if i == 0:
            pdf.set_font('Helvetica', 'B', 12)
        else:
            pdf.set_font('Helvetica', '', 12)
        pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
        for j, value in enumerate(row):
            pdf.cell(table_col_widths[j], table_line_height, value, border=1)

    for pat in tqdm(pats, leave=False):
        # Skip if patient doesn't have region.
        patient = set.patient(pat)
        if not patient.has_region(region):
            continue

        # Start info section.
        pdf.add_page()
        pdf.start_section(pat)
        pdf.start_section('Info', level=1)

        # Add table.
        table_t_margin = 45
        table_l_margin = 12
        table_line_height = 2 * pdf.font_size
        table_col_widths = (40, 40)
        table_data = [('Metric', 'Value')]
        pat_eval_df = eval_df[eval_df['patient-id'] == pat]
        for _, row in pat_eval_df.iterrows():
            table_data.append((
                row.metric,
                f'{row.value:.3f}'
            ))
        for i, row in enumerate(table_data):
            if i == 0:
                pdf.set_font('Helvetica', 'B', 12)
            else:
                pdf.set_font('Helvetica', '', 12)
            pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
            for j, value in enumerate(row):
                pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add images.
        pdf.add_page()
        pdf.start_section('Images', level=1)

        # Save images.
        views = ['axial', 'coronal', 'sagittal']
        img_coords = (
            (img_l_margin, img_t_margin),
            (img_l_margin + img_width, img_t_margin),
            (img_l_margin, img_t_margin + img_height)
        )
        for view, page_coord in zip(views, img_coords):
            # Set figure.
            plot_patient_localiser_prediction(dataset, pat, region, localiser, centre_of=region, show_extent=True, show_patch=True, view=view, window=(3000, 500))

            # Save temp file.
            filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plt.savefig(filepath)
            plt.close()

            # Add image to report.
            pdf.image(filepath, *page_coord, w=img_width, h=img_height)

            # Delete temp file.
            os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'localiser-figures', *localiser, f'{region}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_segmenter_figures(
    dataset: str,
    regions: List[str],
    segmenters: List[Tuple[str, str, str]]) -> None:
    assert len(regions) == len(segmenters)

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(region=regions)

    # Filter regions that don't exist in dataset.
    pat_regions = list(sorted(set.list_regions().region.unique()))
    regions = [r for r in pat_regions if r in regions]

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    logging.info(f"Creating segmenter figures for dataset '{dataset}', regions '{regions}'...")
    for region, segmenter in tqdm(list(zip(regions, segmenters))):
        # Create PDF.
        pdf = FPDF()
        pdf.set_section_title_styles(
            TitleStyle(
                font_family='Times',
                font_style='B',
                font_size_pt=24,
                color=0,
                t_margin=3,
                l_margin=12,
                b_margin=0
            ),
            TitleStyle(
                font_family='Times',
                font_style='B',
                font_size_pt=18,
                color=0,
                t_margin=12,
                l_margin=12,
                b_margin=0
            )
        ) 
        
        # Get errors for the region based upon 'extent-dist-x/y/z' metrics.
        eval_df = load_segmenter_evaluation(dataset, segmenter)

        for pat in tqdm(pats, leave=False):
            # Skip if patient doesn't have region.
            patient = set.patient(pat)
            if not patient.has_region(region):
                continue

            # Start info section.
            pdf.add_page()
            pdf.start_section(pat)
            pdf.start_section('Info', level=1)

            # Add table.
            table_t_margin = 45
            table_l_margin = 12
            table_line_height = 2 * pdf.font_size
            table_col_widths = (40, 40)
            table_data = [('Metric', 'Value')]
            pat_eval_df = eval_df[eval_df['patient-id'] == pat]
            for _, row in pat_eval_df.iterrows():
                table_data.append((
                    row.metric,
                    f'{row.value:.3f}'
                ))
            for i, row in enumerate(table_data):
                if i == 0:
                    pdf.set_font('Helvetica', 'B', 12)
                else:
                    pdf.set_font('Helvetica', '', 12)
                pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
                for j, value in enumerate(row):
                    pdf.cell(table_col_widths[j], table_line_height, value, border=1)

            # Add images.
            pdf.add_page()
            pdf.start_section('Images', level=1)

            # Save images.
            views = ['axial', 'coronal', 'sagittal']
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Set figure.
                plot_patient_segmenter_prediction(dataset, pat, region, segmenter, centre_of=region, view=view, window=(3000, 500))

                # Save temp file.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plt.savefig(filepath)
                plt.close()

                # Add image to report.
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                # Delete temp file.
                os.remove(filepath)

        # Save PDF.
        filepath = os.path.join(set.path, 'reports', 'segmenter-figures', f'{region}.pdf') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.output(filepath, 'F')
