import matplotlib.pyplot as plt
import torch
from typing import *

class Reporter:
    def add_figure(self,
        input_data: torch.Tensor,
        label_data: torch.Tensor,
        prediction_data: torch.Tensor,
        train_step: int,
        step: int,
        sample_index: int,
        axis: str,
        class_labels: dict) -> None:
        """
        effect: adds a figure.
        args:
            input_data: the 3D input data, e.g. CT volume.
            label_data: the 3D binary label.
            prediction_data: the 3D binary prediction.
            train_step: the current training step.
            step: the current visual validation batch/step.
            sample_index: the index of the sample within the batch.
            axis: the axis of view.
            class_labels: the map of predictions to class labels.
        """
        raise NotImplementedError("Method 'add_figure' not implemented in subclass.")

    def add_metric(self,
        tag: str,
        value: float,
        step: int) -> None:
        """
        effect: adds a metric.
        args:
            tag: the metric tag.
            value: the metric value.
            step: the training step - to track changes to the metric over time.
        """
        raise NotImplementedError("Method 'add_metric' not implemented in subclass.")

    def add_model_graph(self,
        model: torch.nn.Module,
        input: torch.Tensor) -> None:
        """
        effect: adds the model graph and layer dimensions.
        args:
            model: the model to record.
            input: a sample input.
        """
        raise NotImplementedError("Method 'add_model_graph' not implemented in subclass.")

    def add_hyperparameters(self,
        params: dict) -> None:
        """
        effect: adds the parameters used for training.
        args:
            params: the parameters used for training.
        """
        raise NotImplementedError("Method 'add_training_params' not implemented in subclass.")
    