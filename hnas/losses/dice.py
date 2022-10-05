import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional

class DiceLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        weights: Optional[List[float]] = None):
        """
        kwargs:
            epsilon: small value to ensure we don't get division by zero.
            weights: the class weights, must sum to one.
        """
        super(DiceLoss, self).__init__()
        if weights is not None:
            assert np.sum(weights) == 1
        self._epsilon = epsilon
        self._weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> float:
        """
        returns: the dice loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the the B x X x Y x Z batch of binary labels.
        """
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # 'torch.argmax' isn't differentiable, so convert label to one-hot encoding
        # and calculate dice per-class/channel.
        label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
        label = F.one_hot(label, num_classes=2)
        label = label.movedim(-1, 1)
        if label.shape != pred.shape:
            raise ValueError(f"DiceLoss expects prediction shape and label shape (after one-hot, dim=1) to be equal. Got '{pred.shape}' and '{label.shape}'.")

        # Flatten volumetric data.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute dice coefficient.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        dice = (2. * intersection + self._epsilon) / (denominator + self._epsilon)

        # Convert dice coef. to dice loss (larger is worse).
        # For dice metric, larger values are worse.
        loss = -dice

        # Determine weights.
        if self._weights is not None:
            if len(self._weights) != loss.shape[1]:
                raise ValueError(f"DiceLoss expects number of weights equal to number of label classes. Got '{len(self._weights)}' and '{loss.shape[1]}'.")
            weights = torch.Tensor(self._weights).to(loss.device)
            weights = weights.unsqueeze(0).repeat(loss.shape[0], 1)
        else:
            weights = torch.ones_like(loss) / loss.shape[1]

        # Apply weights element-wise and sum for each sample.
        loss = (weights * loss).sum(1)

        # Get average across samples in batch.
        loss = loss.mean()

        return loss
