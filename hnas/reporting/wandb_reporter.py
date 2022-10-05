import matplotlib.pyplot as plt
import os
import torch
from typing import *
import wandb

from hnas import config

from .reporter import Reporter

class WandbReporter(Reporter):
    def __init__(self,
        project_name: str,
        run_name: str,
        offline: bool = False) -> None:
        """
        args:
            project_name: the project name.
            run_name: the name of the training run.
        kwargs:
            offline: create files locally, to be uploaded later.
        """
        # Create file locally.
        if offline:
            os.environ['WANDB_MODE'] = 'offline'

        # Configure wandb.
        wandb.init(dir=config.directories.reports, project=project_name, name=run_name)

    """
    For method doc strings, see parent class.
    """
    def add_figure(self,
        input_data: torch.Tensor,
        label_data: torch.Tensor,
        pred_data: torch.Tensor,
        train_step: int,
        step: int,
        sample_index: int,
        axis: str,
        class_labels: dict) -> None:
        # Create mask data.
        mask_data = {
            'prediction': {
                'mask_data': pred_data.numpy(),
                'class_labels': class_labels
            },
            'label': {
                'mask_data': label_data.numpy(),
                'class_labels': class_labels
            }
        }

        # Create image object.
        image = wandb.Image(input_data.numpy(), masks=mask_data)

        # Log the image.
        tag = f"batch_{step}_sample_{sample_index}_axis_{axis}"
        data = { tag: image }
        wandb.log(data, step=train_step)

    def add_metric(self,
        tag: str,
        value: float,
        step: int) -> None:
        data = {tag: value}
        # Log the metric.
        wandb.log(data, step=step)

    def add_model_graph(self,
        model: torch.nn.Module,
        input: torch.Tensor) -> None: 
        pass

    def add_hyperparameters(self,
        params: dict) -> None:
        # Log the hyperparameters.
        wandb.config.update(params)
