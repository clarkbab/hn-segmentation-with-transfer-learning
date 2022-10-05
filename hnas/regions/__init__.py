from typing import List

from hnas import types

from .colours import to_255, RegionColours
from .dose_constraints import get_dose_constraint
from .limits import RegionLimits, truncate_spine
from .patch_sizes import get_region_patch_size
from .regions import RegionNames
from .tolerances import get_region_tolerance, RegionTolerances

def is_region(name: str) -> bool:
    return name in RegionNames

def region_to_list(region: types.PatientRegions) -> List[str]:
    if type(region) == str:
        if region == 'all':
            return RegionNames
        else:
            if not region in RegionNames:
                raise ValueError(f"'{region}' is not a valid region. Should be one of '{RegionNames}'.")
            return [region]
    else:
        for r in region:
            if not r in RegionNames:
                raise ValueError(f"'{r}' is not a valid region. Should be one of '{RegionNames}'.")
        return region
