import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import *

from hnas import config

from .reporter import Reporter

class TensorboardReporter(Reporter):
    def __init__(self,
        project_name: str,
        run_name: str) -> None:
        """
        args:
            project_name: the project name.
            run_name: the name of the training run.
        """
        # Configure tensorboard.
        log_path = os.path.join(config.directories.tensorboard, project_name, run_name)
        self.reporter = SummaryWriter(log_path)

    """
    For method doc strings, see parent class.
    """
    def add_figure(self,
        tag: str,
        figure: plt.Figure,
        iteration: int) -> None:
        self.reporter.add_figure(tag, figure, global_step=iteration)

    def add_metric(self,
        tag: str,
        value: float,
        iteration: int) -> None:
        self.reporter.add_scalar(tag, value, iteration)

    def add_model_graph(self,
        model: torch.nn.Module,
        input: torch.Tensor) -> None: 
        self.reporter.add_graph(model, input)

    def add_hyperparameters(self,
        params: dict) -> None:
        self.reporter.add_hparams(params, {}, run_name='hparams')
