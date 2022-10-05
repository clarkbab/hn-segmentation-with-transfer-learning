import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union

from hnas import config
from hnas import dataset as ds
from hnas.loaders import Loader
from hnas import logging
from hnas.utils import append_row, encode, load_csv, save_csv

def get_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    n_train: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]

    # Create empty dataframe.
    cols = {
        'region': str,
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': str,
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, ds.get(d, 'training', check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    tl, vl, tsl = Loader.build_loaders(datasets, region, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_train=n_train, shuffle_train=False, test_fold=test_fold)
    loader_names = ['train', 'validate', 'test']

    # Get values for this region.
    for loader_name, loader in zip(loader_names, (tl, vl, tsl)):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                origin_ds, origin_pat_id = dataset_map[dataset].sample(sample_id).origin
                data = {
                    'region': region,
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def create_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    logging.info(f"Creating loader manifest for datasets '{datasets}', region '{region}', n_folds '{n_folds}', test_fold '{test_fold}'.")

    # Get manifest.
    df = get_loader_manifest(datasets, region, check_processed=check_processed, n_folds=n_folds, test_fold=test_fold)

    # Save manifest.
    save_csv(df, 'loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv', index=False, overwrite=True)

def load_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    apply_typing: bool = True,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    df = load_csv('loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv')
    df = df.astype({ 'origin-patient-id': str, 'sample-id': str })
    return df
