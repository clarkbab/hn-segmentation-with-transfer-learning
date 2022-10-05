import math
import numpy as np
from skimage import transform

class RandomResample:
    def __init__(self, range, p=1.0):
        self.range = range
        self.p = p

    def __call__(self, data, label=None, binary=False, info=None):
        """
        returns: the input data.
        args:
            data: the input data.
        kwargs:
            label: the label data.
            binary: indicates that input was binary data. Ignored if label is passed.
            info: optional info that the transform may require.
        """
        # Get deterministic function.
        det_fn = self.deterministic()
        
        # Process result.
        if label is not None:
            data = det_fn(data, binary=False, info=info)
            label = det_fn(label, binary=True, info=info)
            return data, label
        
        data = det_fn(data, binary=binary, info=info) 
        return data

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # Realise randomness.
        applied, stretch = self._realise_randomness()

        # Create function that can be called to produce consistent results.
        def fn(data, binary=False, info=None):
            if applied:
                data = self._resample(data, stretch, binary)

            return data

        return fn

    def _realise_randomness(self):
        """
        returns: all realisations of random processes in the transformation.
        """
        # Determine if rotation is applied.
        applied = True if np.random.binomial(1, self.p) else False

        # Determine stretch.
        stretch = np.random.uniform(*self.range)
        
        return applied, stretch 

    def _resample(self, data, stretch, binary):
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        dtype = data.dtype 

        # Get new resolution.
        new_res = [math.floor(stretch * data.shape[i]) for i in range(2)] 

        # Perform resample.
        if binary:
            data = transform.resize(data, new_res, order=0, preserve_range=True)
        else:
            data = transform.resize(data, new_res, order=3, preserve_range=True)

        # Reset types.
        data = data.astype(dtype)

        return data

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
