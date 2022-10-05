import numpy as np

from hnas import types

# Patch sizes in mm.
class RegionPatchSizes:
    BrachialPlexus_L = (170, 165, 160)
    BrachialPlexus_R = (170, 165, 160)
    # Brain = (170, 205, 180)
    Brain = (512, 512, 250)
    BrainStem = (105, 100, 130)
    Cochlea_L = (70, 65, 60)
    Cochlea_R = (70, 65, 60)
    Lens_L = (55, 50, 50)
    Lens_R = (55, 50, 50)
    Mandible = (180, 165, 160)
    OpticNerve_L = (75, 90, 70)
    OpticNerve_R = (75, 90, 70)
    OralCavity = (180, 170, 150)
    Parotid_L = (120, 125, 160)
    Parotid_R = (120, 125, 160)
    SpinalCord = (85, 210, 330)
    Submandibular_L = (80, 85, 110)
    Submandibular_R = (80, 85, 110)

def get_region_patch_size(
    region: str,
    spacing: types.ImageSpacing3D) -> types.ImageSize3D:
    size_mm = getattr(RegionPatchSizes, region)
    size = tuple(np.round(np.array(size_mm) / spacing).astype(int))
    return size
