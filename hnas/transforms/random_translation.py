import hashlib
import json
import numpy as np
from scipy.ndimage import affine_transform

class RandomTranslation:
    def __init__(self, range_x=None, range_y=None, range_z=None, fill=0, p=1.0):
        """
        kwargs:
            range_x: a (min, max) tuple describing the range of possible x-axis translations.
            range_y: a (min, max) tuple describing the range of possible y-axis translations.
            range_z: a (min, max) tuple describing the range of possible z-axis translations.
            fill: value to use for new pixels.
            p: the probability that the transform is applied.
        """
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.fill = fill
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

        return det_fn(data, binary=binary, info=info) 

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # Realise randomness.
        applied, translations = self.realise_randomness()

        # Create function that can be called to produce consistent results.
        def fn(data, binary=False, info=None):
            if applied:
                data = self.translate(data, translations, binary)

            return data

        return fn

    def realise_randomness(self):
        """
        returns: all realisations of random processes in the transformation.
        """
        # Determine if rotation is applied.
        applied = True if np.random.binomial(1, self.p) else False

        # Determine angles.
        translation_x = np.random.uniform(*self.range_x) if self.range_x is not None else 0
        translation_y = np.random.uniform(*self.range_y) if self.range_y is not None else 0
        translation_z = np.random.uniform(*self.range_z) if self.range_z is not None else 0
        translations = (translation_x, translation_y, translation_z)
        
        return applied, translations 

    def translate(self, data, translations, binary):
        """
        returns: the translated data.
        args:
            data: the data to transform.
            translations: the translations to make on (x, y, z) axes.
            binary: indicates that binary data is being transformed.
        """
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        dtype = data.dtype

        # Apply transformation.
        # TODO: Look into whether we should crop or not.
        if binary: 
            data = affine_transform(data, np.identity(3), offset=translations, order=0, cval=0)
        else:
            data = affine_transform(data, np.identity(3), offset=translations, order=3, cval=self.fill)

        # Reset types.
        data = data.astype(dtype)

        return data

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
