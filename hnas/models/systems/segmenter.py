import numpy as np
import os
import pytorch_lightning as pl
from scipy.ndimage import center_of_mass
import torch
from torch import nn
from torch.optim import SGD
from typing import Dict, List, Optional, OrderedDict, Tuple
import wandb

from hnas import config
from hnas.geometry import get_extent_centre
from hnas import logging
from hnas.losses import DiceLoss
from hnas.metrics import batch_mean_dice, batch_mean_all_distances
from hnas.models import replace_checkpoint_alias
from hnas.models.networks import UNet3D
from hnas.postprocessing import get_batch_largest_cc
from hnas import types

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module = DiceLoss(),
        metrics: List[str] = [],
        pretrained_model: Optional[pl.LightningModule] = None,
        spacing: Optional[types.ImageSpacing3D] = None):
        super().__init__()
        if 'distances' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'distances' metric.")
        self.__distances_delay = 50
        self.__distances_interval = 20
        self.__loss = loss
        self.__max_image_batches = 5
        self.__name = None
        self.__metrics = metrics
        pretrained_model = pretrained_model.network if pretrained_model else None
        self.__network = UNet3D(pretrained_model=pretrained_model)
        self.__spacing = spacing

    @property
    def network(self) -> nn.Module:
        return self.__network

    @property
    def name(self) -> Optional[Tuple[str, str, str]]:
        return self.__name

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        check_epochs: bool = True,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs:
            filepath = os.path.join(config.directories.models, model_name, run_name, 'last.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_samples = run_name.split('-')[-1]
            if n_samples == '5':
                n_epochs = 900
            elif n_samples == '10':
                n_epochs = 450
            elif n_samples == '20':
                n_epochs = 300
            else:
                n_epochs = 150
            if state['epoch'] < n_epochs - 1:
                raise ValueError(f"Can't load segmenter ('{model_name}','{run_name}','{checkpoint}') - hasn't completed {n_epochs} epochs training.")

        # Load model.
        model_name, run_name, checkpoint = replace_checkpoint_alias(model_name, run_name, checkpoint)
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Segmenter '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")

        # Update keys by adding '_Segmenter_' prefix if required.
        checkpoint_data = torch.load(filepath, map_location=torch.device('cpu'))
        pairs = []
        update = False
        for k, v in checkpoint_data['state_dict'].items():
            # Get new key.
            if not k.startswith('_Segmenter_'):
                update = True
                new_key = '_Segmenter_' + k
            else:
                new_key = k

            pairs.append((new_key, v))
        checkpoint_data['state_dict'] = OrderedDict(pairs)
        if update:
            logging.info(f"Updating checkpoint keys for model '{(model_name, run_name, checkpoint)}'.")
            torch.save(checkpoint_data, filepath)

        # Load checkpoint.
        segmenter = Segmenter.load_from_checkpoint(filepath, **kwargs)
        segmenter.__name = (model_name, run_name, checkpoint)
        return segmenter

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(
        self,
        x: torch.Tensor,
        probs: bool = False):
        # Get prediction.
        pred = self.__network(x)
        if probs:
            return pred

        # Apply thresholding.
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool_)
        pred = get_batch_largest_cc(pred)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        _, x, y = batch
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('train/loss', loss, on_epoch=True)

        if 'dice' in self.__metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('train/dice', dice, on_epoch=True)

        # if 'hausdorff' in self.__metrics and self.global_step > self.__hausdorff_delay:
        #     if y_hat.sum() > 0:
        #         hausdorff = batch_mean_hausdorff_distance(y_hat, y, self.__spacing)
        #         self.log('train/hausdorff', hausdorff, on_epoch=True)

        # if 'surface' in self.__metrics and self.global_step > self.__surface_delay:
        #     if y_hat.sum() > 0 and y.sum() > 0:
        #         mean_sd, median_sd, std_sd, max_sd = batch_mean_symmetric_surface_distance(y_hat, y, self.__spacing)
        #         self.log('train/mean-surface', mean_sd, **self.__log_args)
        #         self.log('train/median-surface', median_sd, **self.__log_args)
        #         self.log('train/std-surface', std_sd, **self.__log_args)
        #         self.log('train/max-surface', max_sd, **self.__log_args)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        if not batch:
            raise ValueError(f"Batch is none")
        descs, x, y = batch
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        self.log(f"val/batch/loss/{descs[0]}", loss, on_epoch=False, on_step=True)

        if 'dice' in self.__metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, on_epoch=True, sync_dist=True)
            self.log(f"val/batch/dice/{descs[0]}", dice, on_epoch=False, on_step=True)

        if 'distances' in self.__metrics and self.global_step > self.__distances_delay and self.current_epoch % self.__distances_interval == 0:
            if y_hat.sum() > 0 and y.sum() > 0:
                dists = batch_mean_all_distances(y_hat, y, self.__spacing)
                self.log('val/hd', dists['hd'], **self.__log_args, sync_dist=True)
                self.log('val/hd-95', dists['hd-95'], **self.__log_args, sync_dist=True)
                self.log('val/msd', dists['msd'], **self.__log_args, sync_dist=True)
                self.log(f"val/batch/hd/{descs[0]}", dists['hd'], on_epoch=False, on_step=True)
                self.log(f"val/batch/hd-95/{descs[0]}", dists['hd-95'], **self.__log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/msd/{descs[0]}", dists['msd'], **self.__log_args, on_epoch=False, on_step=True)

        # Log predictions.
        if self.logger:
            class_labels = {
                1: 'foreground'
            }

            for i, desc in enumerate(descs):
                if batch_idx > self.__max_image_batches + 1:
                    break

                # Get images.
                x_vol, y_vol, y_hat_vol = x[i, 0].cpu().numpy(), y[i], y_hat[i]

                # Get centre of extent of ground truth.
                centre = get_extent_centre(y_vol)
                if centre is None:
                    logging.info(f'Empty label, desc: {desc}. Sum: {y_vol.sum()}')
                    continue
                    # raise ValueError(f'Empty label, desc: {desc}. Sum: {y_vol.sum()}')

                for axis, centre_ax in enumerate(centre):
                    # Get slices.
                    slices = tuple([centre_ax if i == axis else slice(0, x_vol.shape[i]) for i in range(0, len(x_vol.shape))])
                    x_img, y_img, y_hat_img = x_vol[slices], y_vol[slices], y_hat_vol[slices]

                    # Fix orientation.
                    if axis == 0 or axis == 1:
                        x_img = np.rot90(x_img)
                        y_img = np.rot90(y_img)
                        y_hat_img = np.rot90(y_hat_img)
                    elif axis == 2:
                        x_img = np.transpose(x_img)
                        y_img = np.transpose(y_img) 
                        y_hat_img = np.transpose(y_hat_img)

                    # Send image.
                    image = wandb.Image(
                        x_img,
                        caption=desc,
                        masks={
                            'ground_truth': {
                                'mask_data': y_img,
                                'class_labels': class_labels
                            },
                            'predictions': {
                                'mask_data': y_hat_img,
                                'class_labels': class_labels
                            }
                        }
                    )
                    title = f'{desc}:axis:{axis}'
                    self.logger.experiment.log({ title: image })
