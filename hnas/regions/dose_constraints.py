from typing import Dict

class RegionDoseContstraints:
    BrachialPlexus_L = { 'max': 72 }
    BrachialPlexus_R = { 'max': 66 }
    Brain = { 'max': 60 }
    BrainStem = { 'max': 54 }
    Cochlea_L = { 'max': 45, 'mean': 35}
    Cochlea_R = { 'max': 45, 'mean': 35}
    Lens_L = { 'max': 10 }
    Lens_R = { 'max': 10 }
    Mandible = { 'max': 72 }
    OpticNerve_L = { 'max': 54 }
    OpticNerve_R = { 'max': 54 }
    OralCavity = { 'mean': 42 }
    Parotid_L = { 'mean': 25 }
    Parotid_R = { 'mean': 25 }
    SpinalCord = { 'max': 45 }
    Submandibular_L = { 'mean': 39 }
    Submandibular_R = { 'mean': 39 }

def get_dose_constraint(region: str) -> Dict[str, int]:
    return getattr(RegionDoseContstraints, region)
