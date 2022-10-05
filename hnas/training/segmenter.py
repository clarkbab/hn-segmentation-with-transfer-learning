from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchio.transforms import RandomAffine
from typing import List, Optional, Tuple, Union

from hnas import config
from hnas import dataset as ds
from hnas.loaders import Loader
from hnas import logging
from hnas.losses import DiceLoss
from hnas.models.systems import Segmenter
from hnas.reporting.loaders import get_loader_manifest
from hnas import types
from hnas.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_segmenter(
    dataset: Union[str, List[str]],
    model_name: str,
    run_name: str,
    region: str,
    lr_find: bool = False,
    n_epochs: int = 150,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    pretrained_model: Optional[types.ModelName] = None,    
    p_val: float = 0.2,
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on dataset '{dataset}' with region '{region}' - pretrained model '{pretrained_model}', resume '{resume}'.")

    # Load datasets.
    datasets = arg_to_list(dataset, str)
    spacing = ds.get(datasets[0], 'training').params['output-spacing']
    for dataset in datasets[1:]:
        # Check for consistent spacing.
        new_spacing = ds.get(dataset, 'training').params['output-spacing']
        if new_spacing != spacing:
            raise ValueError(f'Datasets must have consistent spacing.')

    # Create transforms.
    rotation = (-5, 5)
    translation = (-50, 50)
    scale = (0.8, 1.2)
    transform = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')

    # Create data loaders.
    loaders = Loader.build_loaders(datasets, region, extract_patch=True, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, spacing=spacing, test_fold=test_fold, transform=transform)
    train_loader = loaders[0]
    val_loader = loaders[0]

    # Get loss function.
    loss_fn = DiceLoss()

    # Create model.
    metrics = ['dice']
    if pretrained_model:
        pretrained_model = Segmenter.load(*pretrained_model)
    model = Segmenter(
        loss=loss_fn,
        metrics=metrics,
        pretrained_model=pretrained_model,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        logger.watch(model)   # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    checks_path = os.path.join(config.directories.models, model_name, run_name)
    callbacks = [
        ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=checks_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        # Get the checkpoint path.
        resume_run = resume_run if resume_run is not None else run_name
        logging.info(f'Loading ckpt {model_name}, {resume_run}, {resume_ckpt}')
        ckpt_path = os.path.join(config.directories.models, model_name, resume_run, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=n_gpus if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=2,
        precision=16,
        strategy='ddp')

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
