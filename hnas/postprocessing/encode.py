import numpy as np

def one_hot_encode(a: np.ndarray) -> np.ndarray:
    return (np.arange(a.max()) == a[...,None]-1).astype(bool)