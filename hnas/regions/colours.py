import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from hnas import types

# Define region color map.
palette_tab20 = plt.cm.tab20
palette_tab20b = plt.cm.tab20b
class RegionColours:
    BrachialPlexus_L = palette_tab20(0)
    BrachialPlexus_R = palette_tab20(1)
    Brain = palette_tab20(16)
    BrainStem = palette_tab20(17)
    Cochlea_L = palette_tab20(2)
    Cochlea_R = palette_tab20(3)
    Lens_L = palette_tab20(4)
    Lens_R = palette_tab20(5)
    Mandible = palette_tab20b(14)
    OpticNerve_L = palette_tab20(6)
    OpticNerve_R = palette_tab20(7)
    OralCavity = palette_tab20b(6)
    Parotid_L = palette_tab20(8)
    Parotid_R = palette_tab20(9)
    SpinalCord = palette_tab20b(18)
    Submandibular_L = palette_tab20(12)
    Submandibular_R = palette_tab20(13)

def to_255(colour: types.Colour) -> Tuple[int, int, int]:
    """
    returns: a colour in RGB (0-255) scale.
    args:
        colour: the colour in RGB (0-1) scale.
    """
    # Convert colour scale.
    colour = tuple((255 * np.array(colour)).astype(int))
    return colour