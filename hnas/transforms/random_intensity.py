import hashlib
import json
import numpy as np

class RandomIntensity:
    def __init__(self, lam=None, dist='poisson', p=1.0):
        """
        kwargs:
            p: the probability that the rotation will be applied.
        """
        assert dist == 'poisson'
        assert lam is not None
        self.lam = lam
        self.p = p

    def __call__(self, data, binary=False, info=None):
        """
        returns: the input data.
        args:
            data: the input data.
        kwargs:
            binary: indicates that input was binary data.
            info: optional info that the transform may require.
        """
        # Get deterministic function.
        det_fn = self.deterministic()
        
        # Process result.
        result = det_fn(data, binary=binary, info=info) 
        return result

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # Realise randomness.
        applied, seed = self.realise_randomness()

        # Create function that can be called to produce consistent results.
        def fn(data, binary=False, info=None):
            if applied and not binary:
                data = self.apply_noise(data, seed)

            return data

        return fn

    def realise_randomness(self):
        """
        returns: all realisations of random processes in the transformation.
        args:
            shape: the shape of the noise to generate.
        """
        # Determine if rotation is applied.
        applied = True if np.random.binomial(1, self.p) else False

        # Determine noise.
        # We create a seed instead of a noise with data shape, as we don't
        # know the shape of the data when 'deterministic()' is called.
        seed = np.random.randint(0, 2 ** 32, dtype=np.uint32)
        
        return applied, seed 

    def apply_noise(self, data, seed):
        """
        returns: the rotated data.
        args:
            data: the data to transform.
            seed: a random seed used to generate noise.
        """
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        dtype = data.dtype

        # Add the noise.
        np.random.seed(seed)
        noise = np.random.poisson(lam=self.lam, size=data.shape)
        data = data + noise

        # Reset types.
        data = data.astype(dtype)

        return data

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
