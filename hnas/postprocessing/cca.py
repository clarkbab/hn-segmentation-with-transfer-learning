import numpy as np
from skimage.measure import label   

def get_largest_cc(a: np.ndarray) -> np.ndarray:
    """
    returns: a 3D array with largest connected component only.
    args:
        a: a 3D binary array.
    """
    if a.dtype != np.bool_:
        raise ValueError(f"'get_batch_largest_cc' expected a boolean array, got '{a.dtype}'.")

    # Check that there are some foreground pixels.
    labels = label(a)
    if labels.max() == 0:
        return np.zeros_like(a)
    
    # Calculate largest component.
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_cc

def get_batch_largest_cc(a: np.ndarray) -> np.ndarray:
    """
    returns: a batch of 3D arrays with largest connected component only.
    args:
        a: a 3D binary array.
    """
    if a.dtype != np.bool_:
        raise ValueError(f"'get_batch_largest_cc' expected a boolean array, got '{a.dtype}'.")

    ccs = []
    for data in a:
        cc = get_largest_cc(data)
        ccs.append(cc)
    b = np.stack(ccs, axis=0)
    return b
        