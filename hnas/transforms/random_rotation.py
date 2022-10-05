import hashlib
import json
import numpy as np
from scipy.ndimage import rotate

class RandomRotation:
    def __init__(self, range_x=None, range_y=None, range_z=None, fill=0, p=1.0):
        """
        kwargs:
            range_x: a (min, max) tuple describing angle range in degrees about the axial axis.
            range_y: a (min, max) tuple describing angle range in degrees about coronal axis.
            range_z: a (min, max) tuple describing angle range in degrees about sagittal axis.
            fill: values with which to fill new pixels.
            p: the probability that the rotation will be applied.
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
        
        # Process result.
        result = det_fn(data, binary=binary, info=info) 
        return result

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # Realise randomness.
        applied, angles = self.realise_randomness()

        # Create function that can be called to produce consistent results.
        def fn(data, binary=False, info=None):
            if applied:
                data = self.rotate(data, angles, binary)

            return data

        return fn

    def realise_randomness(self):
        """
        returns: all realisations of random processes in the transformation.
        """
        # Determine if rotation is applied.
        applied = True if np.random.binomial(1, self.p) else False

        # Determine angles.
        angle_x = np.random.uniform(*self.range_x) if self.range_x is not None else None
        angle_y = np.random.uniform(*self.range_y) if self.range_y is not None else None
        angle_z = np.random.uniform(*self.range_z) if self.range_z is not None else None
        angles = (angle_x, angle_y, angle_z)
        
        return applied, angles 

    def rotate(self, data, angles, binary):
        """
        returns: the rotated data.
        args:
            data: the data to transform.
            angles: the angle to rotate around the (x, y, z) axes.
            binary: indicates that binary data is being transformed.
        """
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        dtype = data.dtype

        # Sample the range uniformly.
        axes = [(1, 2), (0, 2), (0, 1)]
        for ax, angle in zip(axes, angles):
            # Skip those axes that don't need rotation.
            if angle is None:
                continue
            
            # Rotate the data.
            if binary:
                data = rotate(data, angle, axes=ax, order=0, cval=0) 
            else:
                data = rotate(data, angle, axes=ax, order=3, cval=self.fill)

        # Reset types.
        data = data.astype(dtype)

        return data

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
