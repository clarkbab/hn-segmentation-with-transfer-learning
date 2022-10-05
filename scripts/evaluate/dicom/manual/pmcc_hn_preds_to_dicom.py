from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
hnas_dir = up(up(up(up(up(filepath)))))
sys.path.append(hnas_dir)
from hnas.evaluation.dataset.dicom import create_dose_evaluation

DATASETS = ('PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC') # Code links from 'training' set to nifti set.
REGIONS = (
    'BrachialPlexus_L', # 0
    'BrachialPlexus_R', # 1
    'Brain',            # 2
    'BrainStem',        # 3
    'Cochlea_L',        # 4
    'Cochlea_R',        # 5
    'Lens_L',           # 6
    'Lens_R',           # 7
    'Mandible',         # 8
    'OpticNerve_L',     # 9
    'OpticNerve_R',     # 10
    'OralCavity',       # 11
    'Parotid_L',        # 12
    'Parotid_R',        # 13
    'SpinalCord',       # 14
    'Submandibular_L',  # 15
    'Submandibular_R'   # 16
)
N_FOLDS = 5
TEST_FOLDS = (0, 1, 2, 3, 4)
TEST_FOLDS = (0,)
N_TRAINS = (5, 10, 20, 50, 100, None)
N_TRAINS = (None,)
MODELS = ('clinical', 'public', 'transfer')
MODELS = ('clinical', 'public')
USE_LOADER_MANIFEST = True
USE_MODEL_MANIFEST = True

for test_fold in TEST_FOLDS:
    for region in REGIONS:
        localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')

        for model in MODELS:
            if model == 'public':
                segmenter = (f'segmenter-{region}', 'public-1gpu-150epochs', 'BEST')
                (DATASETS, region, localiser, segmenter, n_folds=N_FOLDS, test_fold=test_fold, use_loader_manifest=USE_LOADER_MANIFEST, use_model_manifest=USE_MODEL_MANIFEST)
            else:
                for n_train in N_TRAINS:
                    segmenter = (f'segmenter-{region}', f'{model}-fold-{test_fold}-samples-{n_train}', 'BEST')
                    convert_segmenter_predictions_to_dicom_from_loader(DATASETS, region, localiser, segmenter, n_folds=N_FOLDS, test_fold=test_fold, use_loader_manifest=USE_LOADER_MANIFEST, use_model_manifest=USE_MODEL_MANIFEST)
