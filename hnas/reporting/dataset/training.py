from fpdf import FPDF, TitleStyle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.ndimage.measurements import label as label_objects
from tqdm import tqdm
from typing import List, Optional, Union
from uuid import uuid1

from hnas import config
from hnas import dataset as ds
from hnas.geometry import get_extent, get_extent_centre
from hnas.loaders import Loader
from hnas import logging
from hnas.plotting.dataset.training import plot_sample_regions
from hnas.postprocessing import get_object, one_hot_encode
from hnas.regions import RegionNames
from hnas import types
from hnas.utils import append_row, encode

def region_count(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # List regions.
    set = ds.get(dataset, 'training')
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
    count_df = regions_df.groupby(['partition', 'region']).count().rename(columns={'sample-index': 'count'})

    # Add 'p' column.
    count_df = count_df.reset_index()
    total_df = count_df.groupby('region').sum().rename(columns={'count': 'total'})
    count_df = count_df.join(total_df, on='region')
    count_df['p'] = count_df['count'] / count_df['total']
    count_df = count_df.drop(columns='total')
    count_df = count_df.set_index(['partition', 'region'])
    return count_df

def create_region_count_report(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> None:
    # Generate counts report.
    set = ds.get(dataset, 'training')
    count_df = region_count(dataset, clear_cache=clear_cache, region=regions)
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath)

def create_region_figures(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get dataset.
    set = ds.get(dataset, 'training')

    # Get regions.
    if type(regions) == str:
        if regions == 'all':
            regions = list(sorted(set.list_regions().region.unique()))
        else:
            regions = [regions]

    # Filter regions that don't exist in dataset.
    pat_regions = list(sorted(set.list_regions().region.unique()))
    regions = [r for r in pat_regions if r in regions]

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    logging.info(f"Creating region figures for dataset '{dataset}', regions '{regions}'...")
    for region in tqdm(regions):
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
            ),
            TitleStyle(
                font_family='Times',
                font_style='B',
                font_size_pt=12,
                color=0,
                t_margin=16,
                l_margin=12,
                b_margin=0
            )
        ) 

        # Define partitions.
        partitions = ['train', 'validation', 'test']

        for partition in tqdm(partitions, leave=False):
            # Load samples.
            part = set.partition(partition)
            samples = part.list_samples(region=region)
            if len(samples) == 0:
                continue

            # Start partition section.
            pdf.add_page()
            pdf.start_section(f'Partition: {partition}')

            for s in tqdm(samples, leave=False):
                # Load sample.
                sample = part.sample(s)

                # Start info section.
                pdf.add_page()
                pdf.start_section(f'Sample: {s}', level=1)
                pdf.start_section('Info', level=2)

                # Add table.
                table_t_margin = 50
                table_l_margin = 12
                table_cols = 5
                table_line_height = 2 * pdf.font_size
                table_col_widths = (15, 35, 30, 45, 45)
                table_width = 180
                table_data = [('ID', 'Volume [vox]', 'Volume [p]', 'Extent Centre [vox]', 'Extent Width [vox]')]
                obj_df = get_object_summary(dataset, partition, s, region)
                for i, row in obj_df.iterrows():
                    table_data.append((
                        str(i),
                        str(row['volume-vox']),
                        f"{row['volume-p']:.3f}",
                        row['extent-centre-vox'],
                        row['extent-width-vox']
                    ))
                for i, row in enumerate(table_data):
                    if i == 0:
                        pdf.set_font('Helvetica', 'B', 12)
                    else:
                        pdf.set_font('Helvetica', '', 12)
                    pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
                    for j, value in enumerate(row):
                        pdf.cell(table_col_widths[j], table_line_height, value, border=1)

                for i, row in obj_df.iterrows():
                    # Start object section.
                    pdf.add_page()
                    pdf.start_section(f'Object: {i}', level=2)

                    # Save images.
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
                        plot_sample_regions(dataset, partition, s, centre_of=region, colours=['y'], postproc=postproc, region=region, show_extent=True, view=view, window=(3000, 500))

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
    partition: str,
    sample: str,
    region: str) -> pd.DataFrame:
    set = ds.get(dataset, 'training')
    samp = set.partition(partition).sample(sample)
    spacing = eval(set.params.spacing[0])
    label = samp.label(region=region)[region]
    objs, n_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = one_hot_encode(objs)
    
    cols = {
        'extent-centre-vox': str,
        'extent-width-vox': str,
        'volume-mm3': float,
        'volume-p': float,
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
        data['extent-width-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = get_extent_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        n_voxels = obj.sum()
        volume = n_voxels * vox_volume
        data['volume-vox'] = n_voxels
        data['volume-p'] = n_voxels / tot_voxels
        data['volume-mm3'] = volume

        df = df.append(data, ignore_index=True)

    df = df.astype(cols)
    return df

def create_ct_figures(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get dataset.
    set = ds.get(dataset, 'training')

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

    logging.info(f"Creating CT figures for dataset '{dataset}', regions '{regions}'...")
    partitions = ['train', 'validation', 'test']
    for partition in tqdm(partitions):
        # Get patients.
        part = set.partition(partition)
        samples = part.list_samples(region=regions)
        if len(samples) == 0:
            continue

        # Start partition section.
        pdf.add_page()
        pdf.start_section(f'Partition: {partition}')

        for s in tqdm(samples, leave=False):
            # Load sample.
            sample = part.sample(s)
            input = sample.input()

            # Show images.
            pdf.add_page()
            pdf.start_section(f'Sample: {s}', level=1)

            # Save images.
            axes = [2, 1, 0]
            views = ['axial', 'coronal', 'sagittal']
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for axis, view, page_coord in zip(axes, views, img_coords):
                # Set figure.
                slice_idx = int(input.shape[axis] / 2)
                plot_sample_regions(dataset, partition, s, region=None, slice_idx=slice_idx, view=view, window=(3000, 500))

                # Save temp file.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plt.savefig(filepath)
                plt.close()

                # Add image to report.
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                # Delete temp file.
                os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'ct-figures.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')
