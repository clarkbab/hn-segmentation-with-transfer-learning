import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from hnas import dataset as ds
from hnas import logging

from ..dataset import DatasetType
from ..training import recreate

def convert_to_training(
    dataset: str,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_val: float = 0.2,
    random_seed: int = 42):
    # Create dataset.
    other_ds = ds.get(dataset, 'other')
    set = recreate(dataset)

    logging.info(f"Converting 'OtherDataset' dataset '{dataset}' into 'TrainingDataset' '{dataset}'...")

    # Load patients.
    samples = other_ds.list_samples() 

    # Shuffle patients.
    np.random.seed(random_seed) 
    np.random.shuffle(samples)

    # Partition patients - rounding assigns more patients to the test set,
    # unless p_test=0, when these are assigned to the validation set.
    n_train = int(np.floor(p_train * len(samples)))
    if p_test == 0:
        n_validation = len(samples) - n_train
    else:
        n_validation = int(np.floor(p_val * len(samples)))
    train_samples = samples[:n_train]
    validation_samples = samples[n_train:(n_train + n_validation)]
    test_samples = samples[(n_train + n_validation):]
    logging.info(f"Num patients per partition: {len(train_samples)}/{len(validation_samples)}/{len(test_samples)} for train/validation/test.")

    # Write data to each partition.
    partitions = ['train', 'validation', 'test']
    partition_samples = [train_samples, validation_samples, test_samples]
    for partition, samples in zip(partitions, partition_samples):
        logging.info(f"Creating partition '{partition}'...")
        # TODO: implement normalisation.

        # Create partition.
        part = set.create_partition(partition)

        # Write each patient to partition.
        for i, pat in enumerate(tqdm(samples, leave=False)):
            # Load data.
            sample = other_ds.sample(pat)
            input = sample.data()

            # Save input data.
            _create_training_input(part, pat, i, input)

    # Indicate success.
    _indicate_success(set, '__CONVERT_TO_TRAINING_SUCCESS__')

def _indicate_success(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def _create_training_input(
    partition: 'TrainingPartition',
    sample_id: int,
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(partition.path, 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

    # Append to manifest.
    manifest_path = os.path.join(partition.dataset.path, 'manifest.csv')
    if not os.path.exists(manifest_path):
        with open(manifest_path, 'w') as f:
            f.write('partition,sample-id,index\n')

    # Append line to manifest. 
    with open(manifest_path, 'a') as f:
        f.write(f"{partition.name},{sample_id},{index}\n")
