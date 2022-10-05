import os
import pandas as pd
from tqdm import tqdm
from typing import List

from hnas import config
from hnas.loaders import Loader
from hnas.regions import RegionNames
from hnas.utils import append_row, load_csv, save_csv

def create_model_manifest() -> None:
    datasets = ('PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC')
    model_types = ['localiser', 'segmenter']
    model_subtypes = ['clinical', 'public', 'transfer']
    n_folds = 5
    n_trains = (5, 10, 20, 50, 100, 200, None)
    regions = RegionNames
    test_folds = tuple(range(n_folds))

    cols = {
        'name': str,
        'run': str,
        'checkpoint': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add public models.
    for model_type in tqdm(model_types):
        for region in tqdm(regions, leave=False):
            name = f'{model_type}-{region}'
            for model_subtype in model_subtypes:
                if model_subtype == 'public':
                    run = 'public-1gpu-150epochs'
                    ckpts = list_checkpoints(name, run)
                    for ckpt in ckpts:
                        data = {
                            'name': name,
                            'run': run,
                            'checkpoint': ckpt
                        }
                        df = append_row(df, data)
                elif model_type == 'segmenter':
                    for test_fold in test_folds:
                        for n_train in n_trains:
                            # Check model exists.
                            tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
                            n_train_max = len(tl) + len(vl)
                            if n_train != None and n_train > n_train_max:
                                continue

                            run = f'{model_subtype}-fold-{test_fold}-samples-{n_train}'
                            ckpts = list_checkpoints(name, run)
                            for ckpt in ckpts:
                                data = {
                                    'name': name,
                                    'run': run,
                                    'checkpoint': ckpt
                                }
                                df = append_row(df, data)

    # Save manifest.
    df = df.astype(cols)
    save_csv(df, 'model-manifest.csv', overwrite=True) 
    
def load_model_manifest():
    return load_csv('model-manifest.csv')

def list_checkpoints(
    name: str,
    run: str) -> List[str]:
    ckptspath = os.path.join(config.directories.models, name, run)
    ckpts = list(sorted([c.replace('.ckpt', '') for c in os.listdir(ckptspath)]))
    return ckpts
