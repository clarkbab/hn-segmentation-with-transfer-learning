import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from ..prediction import get_localiser_prediction
from hnas import dataset as ds
from hnas import logging
from hnas.models.systems import Localiser
from hnas import types

def get_sample_localiser_prediction(
    dataset: str,
    sample_idx: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None) -> None:
    # Load data.
    set = ds.get(dataset, 'training')
    sample = set.sample(sample_idx)
    input = sample.input
    spacing = sample.spacing

    # Make prediction.
    pred = get_localiser_prediction(localiser, loc_size, loc_spacing, input, spacing, device=device)

    return pred
