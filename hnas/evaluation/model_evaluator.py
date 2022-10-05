import logging
import numpy as np
import os
import torch
from torchio import LabelMap, Subject
from tqdm import tqdm

from hnas import config
from hnas.metrics import batch_mean_dice, batch_mean_hausdorff_distance
from hnas import plotting
from hnas.postprocessing import batch_largest_connected_component
from hnas import utils

class ModelEvaluator:
    def __init__(self, run_name, test_loader, device=torch.device('cpu'), metrics=('dice', 'hausdorff'), mixed_precision=True, output_spacing=None,
        output_transform=None, print_interval='epoch', record=True, save_data=False):
        """
        args:
            run_name: the name of the run.
            test_loader: the loader for the test data.
        kwargs:
            device: the device to train on.
            metrics: the metrics to calculate.
            mixed_precision: whether to use PyTorch mixed precision training.
            output_spacing: the voxel spacing of the input data.
            output_transform: the transform to apply before comparing prediction to label.
            print_interval: the interval in which to print the results.
            record: whether to save figures, predictions, labels, etc. or not.
            save_data: whether to save predictions, figures, etc. to disk.
        """
        self.device = device
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.output_spacing = output_spacing
        self.output_transform = output_transform
        if output_transform:
            assert output_spacing, 'Output spacing must be specified if output transform applied.'
        self.print_interval = len(test_loader) if print_interval == 'epoch' else print_interval
        self.record = record
        self.run_name = run_name
        self.test_loader = test_loader

        # Initialise running scores.
        self.running_scores = {}
        keys = ['print', 'total']
        for key in keys:
            self.running_scores[key] = {}
            self.reset_running_scores(key)

    def __call__(self, model):
        """
        effect: prints the model evaluation results.
        args:
            model: the model to evaluate.
        """
        # Put model in evaluation mode.
        model.eval()

        for batch, (input, label, input_raw, label_raw) in enumerate(self.test_loader):
            # Convert input and label.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            pred = model(input)

            # Move data back to cpu for calculations.
            pred, label = pred.cpu(), label.cpu()

            # Convert prediction into binary values.
            pred = pred.argmax(axis=1)

            # Save output predictions and labels.
            if self.output_transform and self.record:
                folder = 'output' if self.output_transform else 'raw'
                filepath = os.path.join(config.directories.evaluation, self.run_name, 'predictions', folder, f"batch-{batch}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.save(filepath, pred.numpy().astype(np.bool_))
                filepath = os.path.join(config.directories.evaluation, self.run_name, 'labels', folder, f"batch-{batch}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.save(filepath, label.numpy().astype(np.bool_))

            # Calculate output metrics.
            if self.output_transform:
                if 'dice' in self.metrics:
                    dice = batch_mean_dice(pred, label)
                    self.running_scores['print']['output-dice'] += dice.item()
                    self.running_scores['total']['output-dice'] += dice.item()

                if 'hausdorff' in self.metrics:
                    hausdorff = batch_mean_hausdorff_distance(pred, label, spacing=self.output_spacing)
                    self.running_scores['print']['output-hausdorff'] += hausdorff.item()
                    self.running_scores['total']['output-hausdorff'] += hausdorff.item()

            # Save output prediction plots.
            if self.output_transform and self.record:
                views = ('sagittal', 'coronal', 'axial')
                for view in views:
                    # Find central slices.
                    centroids = utils.get_batch_centroids(label, view) 

                    # Create and save figures.
                    fig = plotting.plot_batch(input, centroids, figsize=(12, 12), label=label, pred=pred, view=view, return_figure=True)
                    filepath = os.path.join(config.directories.evaluation, self.run_name, 'figures', 'output', f"batch-{batch}-{view}.png")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    fig.savefig(filepath)

            # Transform prediction before comparing to label.
            if self.output_transform:
                # Create torchio 'subject'.
                affine = np.array([
                    [self.output_spacing[0], 0, 0, 0],
                    [0, self.output_spacing[1], 0, 0],
                    [0, 0, self.output_spacing[2], 1],
                    [0, 0, 0, 1]
                ])
                pred = LabelMap(tensor=pred, affine=affine)
                subject = Subject(label=pred)

                # Transform the subject.
                output = self.output_transform(subject)

                # Extract results.
                pred = output['label'].data

                # Save transformed predictions and labels.
                if self.record:
                    filepath = os.path.join(config.directories.evaluation, self.run_name, 'predictions', 'raw', f"batch-{batch}")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    np.save(filepath, pred.numpy().astype(bool))
                    filepath = os.path.join(config.directories.evaluation, self.run_name, 'labels', 'raw', f"batch-{batch}")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    np.save(filepath, label_raw.numpy().astype(bool))

            # Save prediction plots.
            if self.record:
                views = ('sagittal', 'coronal', 'axial')
                for view in views:
                    # Find central slices.
                    centroids = utils.get_batch_centroids(label_raw, view) 

                    # Create and save figures.
                    fig = plotting.plot_batch(input_raw, centroids, figsize=(12, 12), label=label_raw, pred=pred, view=view, return_figure=True)
                    filepath = os.path.join(config.directories.evaluation, self.run_name, 'figures', 'raw', f"batch-{batch:0{config.formatting.sample_digits}}-{view}.png")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    fig.savefig(filepath)

            # Calculate metrics.
            if 'dice' in self.metrics:
                dice = batch_mean_dice(pred, label_raw)
                self.running_scores['print']['dice'] += dice.item()
                self.running_scores['total']['dice'] += dice.item()

            if 'hausdorff' in self.metrics:
                hausdorff = batch_mean_hausdorff_distance(pred, label_raw, spacing=self.output_spacing)
                self.running_scores['print']['hausdorff'] += hausdorff.item()
                self.running_scores['total']['hausdorff'] += hausdorff.item()

            # Print metric results.
            if self.should_print(batch):
                self.print_results(batch)
                self.reset_running_scores('print')

        # Print the averaged results.
        self.print_final_results()

    def should_print(self, batch):
        """
        returns: true if the metric results should be printed, else false.
        args:
            batch: the batch number.
        """
        if (batch + 1) % self.print_interval == 0:
            return True
        else:
            return False

    def print_results(self, batch):
        """
        effect: logs metrics to STDOUT.
        args:
            batch: the batch number.
        """
        message = f"Evaluation - [{batch}]"

        if 'dice' in self.metrics:
            if self.output_transform:
                output_dice = self.running_scores['print']['output-dice'] / self.print_interval
                message += f", Output DSC: {output_dice:{config.formatting.metrics}}"
            dice = self.running_scores['print']['dice'] / self.print_interval
            message += f", DSC: {dice:{config.formatting.metrics}}"

        if 'hausdorff' in self.metrics:
            if self.output_transform:
                output_hd = self.running_scores['print']['output-hausdorff'] / self.print_interval
                message += f", Output HD: {output_hd:{config.formatting.metrics}}"
            hd = self.running_scores['print']['hausdorff'] / self.print_interval
            message += f", HD: {hd:{config.formatting.metrics}}"

        logging.info(message)

    def print_final_results(self):
        """
        effect: logs averaged metrics to STDOUT.
        """
        message = 'Evaluation [mean]'

        if 'dice' in self.metrics:
            if self.output_transform is not None:
                mean_output_dice = self.running_scores['total']['output-dice'] / len(self.test_loader)
                message += f", Mean output DSC={mean_output_dice:{config.formatting.metrics}}"
            mean_dice = self.running_scores['total']['dice'] / len(self.test_loader)
            message += f", Mean DSC={mean_dice:{config.formatting.metrics}}"

        if 'hausdorff' in self.metrics:
            if self.output_transform is not None:
                mean_output_hausdorff = self.running_scores['total']['output-hausdorff'] / len(self.test_loader)
                message += f", Mean output HD={mean_output_hausdorff:{config.formatting.metrics}}"
            mean_hausdorff = self.running_scores['total']['hausdorff'] / len(self.test_loader)
            message += f", Mean HD={mean_hausdorff:{config.formatting.metrics}}"

        logging.info(message)

    def reset_running_scores(self, key):
        """
        effect: initialises the metrics under the key namespace.
        args:
            key: the metric namespace, e.g. print, record, etc.
        """
        if 'dice' in self.metrics:
            self.running_scores[key]['dice'] = 0
            if self.output_transform:
                self.running_scores[key]['output-dice'] = 0
        if 'hausdorff' in self.metrics:
            self.running_scores[key]['hausdorff'] = 0
            if self.output_transform:
                self.running_scores[key]['output-hausdorff'] = 0
