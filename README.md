# HN Segmentation with Transfer Learning

This repository contains the source code used in the study: "Transfer learning for auto-segmentation of 17 organs-at-risk in the head and neck: bridging the gap between institutional and public datasets" (**link to paper**). This paper was published in **link to journal and edition**.

## Installation

1. Install python (v3.8.2).

2. Create and activate virtual environment (example uses python native `venv` module).
```bash
$ python -m venv ~/venvs/transfer-learning
$ source ~/venvs/transfer-learning/bin/activate
```

2. Install python packages.
```bash
$ pip install -r requirements.txt
```

3. Set `HNAS_CODE`/`HNAS_DATA` folder paths. `HNAS_CODE` points to the folder you downloaded this reposity to. `HNAS_DATA` contains all training data, models, reports, etc. will be stored here.
```bash
$ export HNAS_CODE=<code-dir>
$ export HNAS_DATA=<data-dir>
```

## Experiment

<div style="text-align: justify">
The preferred method for running the following steps is using a Slurm-managed computing cluster with GPU capability. The provided [`.slurm` files](scripts/slurm) allow for creation of Slurm jobs with minimal editing. [Python scripts](scripts/python) can be run on any platform with minimial editing.
</div>


> **_NOTE:_**  Note that all scripts should be run from the root project folder (e.g. `python scripts/slurm/step_4/create_jobs.py`).

### Experiment - Steps

If using pretrained models, skip to step...
1. Create folders for public datasets:
```bash
$ cd $HNAS_DATA
$ mkdir -p datasets/dicom; cd datasets/dicom
$ mkdir -p HN1/data HNPCT/data HNSCC/data OPC/data
```

2. Download public datasets from the Cancer Imaging Archive to their respective `data` folders. E.g. download `Head-Neck-Radiomics-HN1` to `$HNAS_DATA/datasets/dicom/HN1/data`. 
- [Head-Neck-Radiomics-HN1](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-Radiomics-HN1)
- [Head-Neck-PET-CT](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT)
- [HNSCC](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)
- [OPC-Radiomics](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948764)

3. Public datasets use different names for the same OAR and we need to map these to consistent names. This mapping uses a `region-map.csv` file in the root direction of the dataset. Additionally, there are some region names that are duplicated for a patient and RTSTRUCT (throws an error), we can skip these by creating 'region-dups.csv' files.Symlink the existing 'region-map.csv' and 'region-dups.csv' files using:
```bash
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/hn1-region-map.csv $HNAS_DATA/datasets/dicom/HN1/region-map.csv
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/hnpct-region-map.csv $HNAS_DATA/datasets/dicom/HNPCT/region-map.csv
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/hnpct-region-dups.csv $HNAS_DATA/datasets/dicom/HNPCT/region-dups.csv
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/hnscc-region-map.csv $HNAS_DATA/datasets/dicom/HNSCC/region-map.csv
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/hnscc-region-dups.csv $HNAS_DATA/datasets/dicom/HNSCC/region-dups.csv
$ ln -s $HNAS_CODE/hnas/dataset/dicom/files/region-maps/opc-region-map.csv $HNAS_DATA/datasets/dicom/OPC/region-map.csv
```

4. Create `NIFTI` datasets from DICOM datasets.
```bash
$ python scripts/slurm/steps/4/create_jobs.py           # Runs on slurm cluster - creates 4 jobs.
$ python scripts/python/steps/4.py                      # Runs on local machine.
```

5. Create training data (from `NIFTI` dataset) for public localiser/segmenter models.
```bash
$ python scripts/slurm/steps/5/create_jobs.py           # Runs on slurm cluster - creates 8 jobs.
$ python scripts/python/steps/5.py                      # Runs on local machine.
```

6. Train public localiser/segmenter models per OAR. Note that some editing of Slurm template files will likely be required to connect to your GPU partition.
Training is set up to use [wandb](https://wandb.ai/) logging, but this must be enabled by setting `USE_LOGGER=True` in the templates.
```bash
$ python scripts/slurm/steps/6/create_jobs.py           # Runs on slurm cluster - creates 34 jobs.
$ python scripts/python/steps/6.py                      # Runs on local machine.
```

Training can be resumed upon failure.
```bash
$ python scripts/slurm/steps/6/create_resume_jobs.py        # Runs on slurm cluster - creates 34 jobs.
$ python scripts/python/steps/6_resume.py                   # Runs on local machine.
```

7. Create your institutional dataset (e.g. name='INST') following the [DICOM](#dicom-dataset---setup) or [NIFTI](#nifti-dataset---setup)) setup.

8. (If using DICOM dataset) Process DICOM dataset to NIFTI using the following command:

```bash
$ python scripts/slurm/steps/8/create_job.py            # Runs on slurm cluster - creates 1 job.
$ python scripts/steps/8.py                             # Runs on local machine.
```

9. Create training data (from `NIFTI` dataset) for institutional segmenter models.
```bash
$ python scripts/slurm/steps/9/create_job.py            # Runs on slurm cluster - creates 1 jobs.
$ python scripts/python/steps/9.py                      # Runs on local machine.
```

10. (Note: creates 595 jobs! 17 OARs * 7 sample sizes * 5 folds) Train institutional segmenter model per OAR for increasing sample sizes (e.g. n=5,10,20,...). Note that some editing of Slurm template files will likely be required to connect to your GPU partition.
Training is set up to use [wandb](https://wandb.ai/) logging, but this must be enabled by setting `USE_LOGGER=True` in the templates.
```bash
$ python scripts/slurm/steps/10/create_jobs.py           # Runs on slurm cluster - creates 595 jobs.
$ python scripts/python/steps/10.py                      # Runs on local machine.
```

11. (Note: creates 595 jobs! 17 OARs * 7 sample sizes * 5 folds) This step requires the completion of the public segmenter models (step 6). Train institutional segmenter model per OAR for increasing sample sizes (e.g. n=5,10,20,...). Note that some editing of Slurm template files will likely be required to connect to your GPU partition.
Training is set up to use [wandb](https://wandb.ai/) logging, but this must be enabled by setting `USE_LOGGER=True` in the templates.
```bash
$ python scripts/slurm/steps/11/create_jobs.py           # Runs on slurm cluster - creates 595 jobs.
$ python scripts/python/steps/11.py                      # Runs on local machine.
```

## Datasets

### DICOM Dataset

#### DICOM Dataset - Setup

To add a DICOM dataset, drop all data into the folder `$HNAS_DATA/datasets/dicom/<dataset-name>/data` where `<dataset-name>` is the name of your dataset as it will appear in the `Dataset` API.

Note that *no dataset file structure* is enforced as the indexing engine will traverse the folder, locating all DICOM files, and creating an index (at `.../<dataset-name>/index.csv`) that will be used by the `Dataset` API to make queries on the dataset.

#### DICOM Dataset - Index

The index is built when a dataset is first used via the `Dataset` API. Indexing can also be triggered via the command:

```
from hnas.dataset.dicom import build_index
build_index('<dataset-name>')
```

The index contains a hierarchy of objects that can be queried using the `Dataset` API. During building of the index, some objects may be excluded if they don't meet the inclusion criteria, e.g. a patient will be excluded from the index if they don't have valid CT/RTSTRUCT series. All excluded objects are stored in `.../<dataset-name>/index-errors.csv`.

The index object hierarchy is:

```
- <dataset> 
    - <patient 1>
        - <study 1>
            - <series 1> (e.g. CT)
            - <series 2> (e.g. RTSTRUCT)
        - <study 2>
            ...
    - <patient 2>
        ...
```

##### DICOM Dataset - Index - Exclusion Criteria

The following rules are applied, *in the listed order*, to exclude objects from the index. All excluded objects are saved in `.../<dataset-name>/index-errors.csv` with the applicable error code.

Order | Code | Description
--- | --- | ---
1 | DUPLICATE | Duplicate DICOM files are removed.<br/>Duplicates are determined by DICOM field 'SOPInstanceUID'
2 | NON-STANDARD-ORIENTATION | CT DICOM series with non-standard orientation are removed.<br/>DICOM field 'ImageOrientationPatient' is something other than `[1, 0, 0, 0, 1, 0]`
3 | INCONSISTENT-POSITION-XY | CT DICOM series with inconsistent x/y position across slices are removed.<br/>DICOM field 'ImagePositionPatient' x/y elements should be consistent.
4 | INCONSISTENT-SPACING-XY | CT DICOM series with inconsistent x/y spacing across slices are removed.<br/>DICOM field 'PixelSpacing' should be consistent.
5 | INCONSISTENT-SPACING-Z | CT DICOM series with inconsistent z spacing are removed.<br/>Difference between DICOM field 'ImagePositionPatient' z position for slices (sorted by z position) should be consistent.
6 | MULTIPLE-FILES | Duplicate RTSTRUCT/RTPLAN/RTDOSE files for a series are removed.<br/>First RTSTRUCT/RTPLAN/RTDOSE are retrained for a series (ordered by DICOM field 'SOPInstanceUID')
7 | NO-REF-CT | RTSTRUCT DICOM series without a referenced CT series are removed.<br/>CT series referenced by RTSTRUCT DICOM field `ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID` should be present in the index.
8 | NO-REF-RTSTRUCT | RTPLAN DICOM series without a referenced RTSTRUCT series are removed.<br/>RTSTRUCT series referenced by RTPLAN DICOM field `ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID` should be present in the index.
9 | NO-REF-RTPLAN | RTDOSE DICOM series without a referenced RTPLAN series are removed.<br/>RTPLAN series referenced by RTDOSE DICOM field `ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID` should be present in the index.
10 | NO-RTSTRUCT | Studies without RTSTRUCT series in index are removed.

Feel free to add/remove/update these criteria by modifying the file at `hnas/dataset/dicom/index.py`.

#### DICOM Dataset - Region Maps

You must map your dataset organ-at-risk (OAR) names to [our internal names](hnas/regions/regions.py) using a region map. Region map examples are shown in `hnas/dataset/dicom/files/region-maps`. 

Add the region map at `.../<dataset-name>/region-map.csv`. The map must contain a `dataset` column with region names as they appear in the dataset (regexp capable) and an `internal` column with the internal name to map to. An optional column `case-sensitive` specifies whether the `dataset` column is case sensitive (default=False).

#### DICOM Dataset - API

If there are multiple studies/RTSTRUCTs for a patient, the first study and RTSTRUCT are chosen as the default. To set different defaults, pass `study_index` and `rtstruct_index` kwargs upon patient creation (e.g. `set.patient('<patient-id>', study_index=1, rtstruct_index=2))`. The default CT is always the CT series attached to the default RTSTRUCT series.

##### DICOM Dataset - API - Datasets

```
from hnas import dataset as ds
from hnas.dataset.dicom import DicomDataset

# List datasets.
ds.list_datasets()

# Load dataset.
set = ds.get('<dataset>', 'dicom')
set = ds.get('<dataset>')           # Will raise an error if there are multiple datasets with this name.
set = DicomDataset('<dataset>')     # Using constructor directly.
```

##### DICOM Dataset - API - Patients

```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')

# List patients.
set.list_patients()
set.list_patients(region=['Brain', 'BrainStem'])       # List patients who have certain regions. Slow query as it must read RTSTRUCT file associated with all patients.

# Check for patient.
set.has_patient('<patient-id>')

# Load patient.
pat = set.patient('<patient-id>')

# List patient regions (using default CT/RTSTRUCT series).
pat.list_regions()

# Get CT/region data (using default CT/RTSTRUCT series).
pat.ct_data
pat.region_data(region=['Brain', 'BrainStem'])
```

##### DICOM Dataset - API - Studies
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')

# List studies.
pat.list_studies()

# Load study.
pat.study('<study-id>')
```

##### DICOM Dataset - API - Series
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)

# List series of modality 'ct', 'rtstruct', 'rtplan' or 'rtdose'.
study.list_series('ct')
study.list_series('rtstruct')

# Load series.
study.series('<series-id>')
```

###### DICOM Dataset - API - CT Series
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Load CT data.
series.data

# Load CT geometry.
series.size
series.spacing
series.orientation

# Get pydicom CT 'FileDataset' objects.
series.get_cts()
series.get_first_ct()       # If reading duplicated fields (e.g. PatientName) just load the first one.
```

###### DICOM Dataset - API - RTSTRUCT Series
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# List regions.
series.list_regions()

# Check for region.
series.has_region('Brain')

# Load RTSTRUCT region data.
series.region_data(region=['Brain', 'BrainStem'])

# Get pydicom RTSTRUCT 'FileDataset' object.
series.get_rtstruct()
```

###### DICOM Dataset - API - RTPLAN Series
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Get pydicom RTPLAN 'FileDataset' object.
series.get_rtplan()
```

###### DICOM Dataset - API - RTDOSE Series
```
from hnas import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Load RTDOSE data.
series.data

# Load RTDOSE geometry.
series.size
series.spacing
series.orientation

# Get pydicom RTDOSE 'FileDataset' object.
series.get_rtdose()
```

### NIFTI Dataset

#### NIFTI Dataset - Setup

NIFI datasets can be created by processing an existing DICOM dataset or by manually creating a folder of the correct structure.

##### NIFTI Dataset - Setup - Processing DICOM

We can process the CT and region data from an existing `DICOMDataset` into a `NIFTIDataset` using the following command. We can specify the subset of regions which we'd like to include in our `NIFTIDataset` and also whether to anonymise patient IDs (for transferral to external system for training/evaluation, e.g. high-performance computing cluster).

```
from hnas.processing.dataset.dicom import convert_to_nifti

convert_to_nifti('<dataset-name>', region=['Brain, 'BrainStem'], anonymise=False)
```

When anonymising, a map linking from anonymous patient ID back to the true patient IDs will be saved in `$HNAS_DATA/datasets/nifti/<dataset-name>/index.csv`.

##### NIFTI Dataset - Setup - Manual Creation

NIFTI datasets can be created by adding CT and region NIFTI files to a folder `$HNAS_DATA/datasets/dicom/<dataset-name>/data` with the following structure:

```
<dataset-name> (e.g. 'MyDataset')
    data
        ct
            - <patient 1>.nii (e.g. 0.nii)
            - <patient 2>.nii (e.g. 1.nii)
            - ...
        regions
            <region 1> (e.g. BrachialPlexus_L)
                - <patient 1>.nii
                - <patient 2>.nii
                - ...
            <region 2> (e.g. BrachialPlexus_R)
                - ...
```

#### NIFTI Dataset - API

##### NIFTI Dataset - API - Datasets

```
from hnas import dataset as ds
from hnas.dataset.nifti import NiftiDataset

# List datasets.
ds.list_datasets()

# Load dataset.
set = ds.get('<dataset>', 'nifti')
set = ds.get('<dataset>')           # Will raise an error if there are multiple datasets with this name.
set = NiftiDataset('<dataset>')     # Using constructor directly.
```

##### NIFTI Dataset - API - Patients

```
from hnas import dataset as ds
set = ds.get('<dataset>', 'nifti')

# List patients.
set.list_patients()
set.list_patients(region=['Brain', 'BrainStem'])       # List patients who have certain regions. Fast query as it's just reading filenames.

# Check for patient.
set.has_patient('<patient-id>')

# Load patient.
pat = set.patient('<patient-id>')

# List patient regions.
pat.list_regions()

# Load CT/region data.
pat.ct_data
pat.region_data(region=['Brain', 'BrainStem'])

# Load CT geometry.
pat.size
pat.spacing
pat.offset

# Get de-anonymised patient ID. Anon map must be present.
pat.origin
```

### TRAINING datasets

#### Setup

A `TrainingDataset` must be created by running a processing script on an existing `NiftiDataset`. For example:

```
from hnas.processing.dataset.nifti import convert_to_training

convert_to_training(
    '<nifti-dataset>',              # Source dataset name.
    ['Brain', 'BrainStem'],         # Regions to process.
    '<training-dataset>',           # Target dataset name.
    dilate_iter=3,                  # Number of rounds of dilation to perform to 'dilate_regions'.
    dilate_region=['BrainStem'],   # Regions to dilate (e.g. for localiser training)
    size=(100, 100, 100),           # Crop processed images/labels to this size.
    spacing=(4, 4, 4)               # Resample images/labels to the spacing.
)
```

### Dataset API

The `Dataset` API allows for basic queries to be performed on the installed datasets.

#### DICOM Dataset


#### NIFTI Dataset

#### TRAINING Dataset

## Visualisation




## Pretrained public models

These models were trained on the following Cancer Imaging Archive (TCIA) datasets:
- HN1
- HNPCT
- HNSCC
- OPC

You can download the pretrained public localiser

## Training public model

