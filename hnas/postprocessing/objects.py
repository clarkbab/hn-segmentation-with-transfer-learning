import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label

from .encode import one_hot_encode

def get_object(
    a: np.ndarray,
    obj: int) -> np.ndarray:
    a, _ = label(a, structure=np.ones((3, 3, 3)))
    a = one_hot_encode(a)
    return a[:, :, :, obj]
