class RandomElasticDeformation:
    def __init__(self):
        return None

    def __call__(self):
        print('calling')

    def deform(self):
        # Generate coarse grid of random vectors.
        # How to generate vectors?
        print('deforming')

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")
    