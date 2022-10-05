import numpy as np
from typing import Tuple

def translate(
    a: np.ndarray,
    t: Tuple[int, int, int]) -> np.ndarray:
    return np.roll(a, t, axis=list(range(len(t))))
