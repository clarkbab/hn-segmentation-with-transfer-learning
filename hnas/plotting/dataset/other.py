import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, Union

from hnas import dataset as ds

def plot_sample(
    dataset: str,
    sample_id: Union[int, str],
    figsize: Tuple[int, int] = (12, 12)) -> None:
    # Get data.
    set = ds.get(dataset, 'other')
    filepath = os.path.join(set.path, 'data', f'{sample_id}.npz')
    data = np.load(filepath)['data']

    plt.figure(figsize=figsize)
    plt.imshow(data)
