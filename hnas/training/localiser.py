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
from hnas.losses import DiceLoss
from hnas import logging
from hnas.models.systems import Localiser
from hnas.utils import arg_to_list

def train_localiser(
    dataset: Union[str, List[str]],
    model_name: str,
    run_name: str,
    region: str,
    n_epochs: int = 150,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    pretrained: Optional[Tuple[str, str, str]] = None,
    p_val: float = 0.2,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on dataset '{dataset}' with region '{region}' using '{n_folds}' folds with test fold '{test_fold}'.")

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
    loaders = Loader.build_loaders(datasets, region, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, spacing=spacing, test_fold=test_fold, transform=transform)
    train_loader = loaders[0]
    val_loader = loaders[0]

    # Get loss function.
    loss_fn = DiceLoss()

    # Create model.
    if pretrained:
        pretrained = Localiser.load(*pretrained)
    model = Localiser(
        loss=loss_fn,
        metrics=['dice'],
        pretrained=pretrained,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        logger.watch(model) # Caused multi-GPU training to hang.
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
        if resume_checkpoint is None:
            raise ValueError(f"Must pass 'resume_checkpoint' when resuming training run.")
        check_path = os.path.join(checks_path, f"{resume_checkpoint}.ckpt")
        opt_kwargs['resume_from_checkpoint'] = check_path
    
    # Perform training.
    trainer = Trainer(
        callbacks=callbacks,
        gpus=list(range(n_gpus)),
        logger=logger,
        max_epochs=n_epochs,
        num_sanity_val_steps=0,
        precision=16,
        **opt_kwargs)
    trainer.fit(model, train_loader, val_loader)
