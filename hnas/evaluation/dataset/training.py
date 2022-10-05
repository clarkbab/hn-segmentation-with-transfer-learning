import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Tuple, Union

from hnas import cache
from hnas import dataset as ds
from hnas.metrics import dice, distances
from hnas.models.systems import Localiser, Segmenter
from hnas import logging
from hnas.prediction.dataset.training import get_localiser_prediction
from hnas import types

def evaluate_localiser_predictions(
    dataset: str,
    partitions: Union[str, List[str]],
    localiser: Tuple[str, str, str],
    region: str) -> None:
    logging.info(f"Evaluating predictions for dataset '{dataset}', partitions '{partitions}', region '{region}' using localiser '{localiser}'.")

    # Load dataset.
    set = ds.get(dataset, 'training')

    # Convert partitions arg to list.
    if type(partitions) == str:
        partitions = [partitions]
    
    for partition in partitions:
        # Load samples.
        samples = set.partition(partition).list_samples(region=region)

        # Create dataframe.
        cols = {
            'sample-id': str,
            'metric': str,
            region: float
        }
        df = pd.DataFrame(columns=cols.keys())

        for sample in tqdm(samples):
            # Get pred/ground truth.
            pred = get_localiser_prediction(dataset, partition, sample, localiser)
            label = set.partition(partition).sample(sample).label(region=region)[region].astype(np.bool_)

            # Add metrics.
            metrics = [
                'dice',
                'assd',
                'surface-hd',
                'surface-95hd',
                'voxel-hd',
                'voxel-95hd'
            ]
            data = {}
            for metric in metrics:
                data[metric] = {
                    'sample-id': sample,
                    'metric': metric
                }

            # Dice.
            dsc = dice(pred, label)
            data['dice'][region] = dsc
            df = append_row(df, data['dice'])

            # Distances.
            spacing = eval(set.params.spacing[0])
            try:
                dists = distances(pred, label, spacing)
            except ValueError:
                dists = {
                    'assd': np.nan,
                    'surface-hd': np.nan,
                    'surface-95hd': np.nan,
                    'voxel-hd': np.nan,
                    'voxel-95hd': np.nan
                }

            data['assd'][region] = dists['assd']
            data['surface-hd'][region] = dists['surface-hd']
            data['surface-95hd'][region] = dists['surface-95hd']
            data['voxel-hd'][region] = dists['voxel-hd']
            data['voxel-95hd'][region] = dists['voxel-95hd']
            df = df.append(data['assd'], ignore_index=True)
            df = df.append(data['surface-hd'], ignore_index=True)
            df = df.append(data['surface-95hd'], ignore_index=True)
            df = df.append(data['voxel-hd'], ignore_index=True)
            df = df.append(data['voxel-95hd'], ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        # Set index.
        df = df.set_index('sample-id')

        # Save evaluation.
        filepath = os.path.join(set.path, 'evaluation', partition, 'localiser', *localiser, 'eval.csv') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def get_localiser_evaluation(
    dataset: str,
    partition: str,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'evaluation', partition, 'localiser', *localiser, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Evaluation for dataset '{set}', partition '{partition}', localiser '{localiser}' not found.")
    data = pd.read_csv(filepath)
    return data
