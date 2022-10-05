from ast import literal_eval
from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom as dcm
import os
from time import time
from tqdm import tqdm
from typing import Dict, List

from hnas import config
from hnas import logging
from hnas.utils import append_dataframe, append_row

INDEX_COLS = {
    'patient-id': str,
    'study-id': str,
    'modality': str,
    'series-id': str,
    'sop-id': str,
    'filepath': str,
    'mod-spec': object
}
ERRORS_COLS = INDEX_COLS.copy()
ERRORS_COLS['error'] = str

def build_index(
    dataset: str,
    from_temp_index: bool = False) -> None:
    start = time()

    # Load dataset path.
    dataset_path = os.path.join(config.directories.datasets, 'dicom', dataset) 

    # Create index.
    index = pd.DataFrame(columns=INDEX_COLS.keys())

    # Crawl folder structure.
    temp_filepath = os.path.join(config.directories.temp, f'{dataset}-index.csv')
    if from_temp_index:
        if os.path.exists(temp_filepath):
            logging.info(f"Loading saved index for dataset '{dataset}'...")
            index = pd.read_csv(temp_filepath, index_col='index')
            index['mod-spec'] = index['mod-spec'].apply(lambda m: literal_eval(m))      # Convert str to dict.
        else:
            raise ValueError(f"Temporary index doesn't exist for dataset '{dataset}' at filepath '{temp_filepath}'.")
    else:
        data_path = os.path.join(dataset_path, 'data')
        if not os.path.exists(data_path):
            raise ValueError(f"No 'data' folder found for dataset '{dataset}'.")

        # Add all DICOM files.
        logging.info(f"Building index for dataset '{dataset}'...")
        file_index = 0
        for root, _, files in tqdm(os.walk(data_path)):
            for f in files:
                # Check if DICOM file.
                filepath = os.path.join(root, f)
                try:
                    dicom = dcm.read_file(filepath, stop_before_pixels=True)
                except dcm.errors.InvalidDicomError:
                    continue

                # Get modality.
                modality = dicom.Modality
                if not modality in ('CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE'):
                    continue

                # Get patient ID.
                pat_id = dicom.PatientID

                # Get study UID.
                study_id = dicom.StudyInstanceUID

                # Get series UID.
                series_id = dicom.SeriesInstanceUID

                # Get SOP UID.
                sop_id = dicom.SOPInstanceUID

                # Get modality-specific info.
                if modality == 'CT':
                    if not hasattr(dicom, 'ImageOrientationPatient'):
                        logging.error(f"No 'ImageOrientationPatient' found for CT dicom '{filepath}'.")
                        continue

                    mod_spec = {
                        'ImageOrientationPatient': dicom.ImageOrientationPatient,
                        'ImagePositionPatient': dicom.ImagePositionPatient,
                        'InstanceNumber': dicom.InstanceNumber,
                        'PixelSpacing': dicom.PixelSpacing
                    }
                elif modality == 'RTDOSE':
                    mod_spec = {
                        'RefRTPLANSOPInstanceUID': dicom.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                    }
                elif modality == 'RTPLAN':
                    mod_spec = {
                        'RefRTSTRUCTSOPInstanceUID': dicom.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                    }
                elif modality == 'RTSTRUCT':
                    mod_spec = {
                        'RefCTSeriesInstanceUID': dicom.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    }

                # Add index entry.
                data = {
                    'patient-id': pat_id,
                    'study-id': study_id,
                    'modality': modality,
                    'series-id': series_id,
                    'sop-id': sop_id,
                    'filepath': filepath,
                    'mod-spec': mod_spec,
                }
                index = append_row(index, data, index=file_index)
                file_index += 1
    
        # Save index - in case something goes wrong later.
        index.to_csv(temp_filepath, index=True)

    # Create errors index.
    errors = pd.DataFrame(columns=ERRORS_COLS.keys())

    # Remove duplicates by 'SOPInstanceUID'.
    logging.info(f"Removing duplicate DICOM files (by 'SOPInstanceUID')...")
    dup_rows = index['sop-id'].duplicated()
    dup = index[dup_rows]
    dup['error'] = 'DUPLICATE'
    errors = append_dataframe(errors, dup)
    index = index[~dup_rows]

    # Check CT slices have standard orientation.
    logging.info(f"Removing CT DICOM files with rotated orientation (by 'ImageOrientationPatient')...")
    ct = index[index.modality == 'CT']
    def standard_orientation(m: Dict) -> bool:
        orient = m['ImageOrientationPatient']
        return orient == [1, 0, 0, 0, 1, 0]
    stand_orient = ct['mod-spec'].apply(standard_orientation)
    nonstand_idx = stand_orient[~stand_orient].index
    nonstand = index.loc[nonstand_idx]
    nonstand['error'] = 'NON-STANDARD-ORIENTATION'
    errors = append_dataframe(errors, nonstand)
    index = index.drop(nonstand_idx)

    # Check CT slices have consistent x/y position.
    logging.info(f"Removing CT DICOM files with inconsistent x/y position (by 'ImagePositionPatient')...")
    ct = index[index.modality == 'CT']
    def consistent_xy_position(series: pd.Series) -> bool:
        pos = series.apply(lambda m: pd.Series(m['ImagePositionPatient'][:2]))
        pos = pos.drop_duplicates()
        return len(pos) == 1
    cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_position)
    incons_idx = cons_xy[~cons_xy].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-POSITION-XY'
    errors = append_dataframe(errors, incons)
    index = index.drop(incons_idx)

    # Check CT slices have consistent x/y spacing.
    logging.info(f"Removing CT DICOM files with inconsistent x/y spacing (by 'PixelSpacing')...")
    ct = index[index.modality == 'CT']
    def consistent_xy_spacing(series: pd.Series) -> bool:
        pos = series.apply(lambda m: pd.Series(m['PixelSpacing']))
        pos = pos.drop_duplicates()
        return len(pos) == 1
    cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_spacing)
    incons_idx = cons_xy[~cons_xy].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-SPACING-XY'
    errors = append_dataframe(errors, incons)
    index = index.drop(incons_idx)

    # Check CT slices have consistent z spacing.
    logging.info(f"Removing CT DICOM files with inconsistent z spacing (by 'ImagePositionPatient')...")
    ct = index[index.modality == 'CT']
    def consistent_z_position(series: pd.Series) -> bool:
        z_locs = series.apply(lambda m: m['ImagePositionPatient'][2]).sort_values()
        z_diffs = z_locs.diff().dropna().round(3)
        z_diffs = z_diffs.drop_duplicates()
        return len(z_diffs) == 1
    cons_z = ct.groupby('series-id')['mod-spec'].transform(consistent_z_position)
    incons_idx = cons_z[~cons_z].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-SPACING-Z'
    errors = append_dataframe(errors, incons)
    index = index.drop(incons_idx)

    # Check that RTSTRUCT references CT series in index.
    logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID')...")
    ct_series = index[index.modality == 'CT']['series-id'].unique()
    rtstruct = index[index.modality == 'RTSTRUCT']
    ref_ct = rtstruct['mod-spec'].apply(lambda m: m['RefCTSeriesInstanceUID']).isin(ct_series)
    nonref_idx = ref_ct[~ref_ct].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-CT'
    errors = append_dataframe(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that RTPLAN references RTSTRUCT SOP instance in index.
    logging.info(f"Removing RTPLAN DICOM files without RTSTRUCT in index (by 'RefRTSTRUCTSOPInstanceUID')...")
    rtstruct_sops = index[index.modality == 'RTSTRUCT']['sop-id'].unique()
    rtplan = index[index.modality == 'RTPLAN']
    ref_rtstruct = rtplan['mod-spec'].apply(lambda m: m['RefRTSTRUCTSOPInstanceUID']).isin(rtstruct_sops)
    nonref_idx = ref_rtstruct[~ref_rtstruct].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-RTSTRUCT'
    errors = append_dataframe(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that RTDOSE references RTPLAN SOP instance in index.
    logging.info(f"Removing RTDOSE DICOM files without RTPLAN in index (by 'RefRTPLANSOPInstanceUID')...")
    rtplan_sops = index[index.modality == 'RTPLAN']['sop-id'].unique()
    rtdose = index[index.modality == 'RTDOSE']
    ref_rtplan = rtdose['mod-spec'].apply(lambda m: m['RefRTPLANSOPInstanceUID']).isin(rtplan_sops)
    nonref_idx = ref_rtplan[~ref_rtplan].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-RTPLAN'
    errors = append_dataframe(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that study has RTSTRUCT series.
    logging.info(f"Removing series without RTSTRUCT DICOM...")
    incl_rows = index.groupby('study-id')['modality'].transform(lambda s: 'RTSTRUCT' in s.unique())
    nonincl = index[~incl_rows]
    nonincl['error'] = 'STUDY-NO-RTSTRUCT'
    errors = append_dataframe(errors, nonincl)
    index = index[incl_rows]

    # Save index.
    if len(index) > 0:
        index = index.astype(INDEX_COLS)
    filepath = os.path.join(dataset_path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Save errors index.
    if len(errors) > 0:
        errors = errors.astype(ERRORS_COLS)
    filepath = os.path.join(dataset_path, 'index-errors.csv')
    errors.to_csv(filepath, index=False)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(dataset_path, f'__INDEXING_TIME_MINS_{mins}__')
    Path(filepath).touch()
