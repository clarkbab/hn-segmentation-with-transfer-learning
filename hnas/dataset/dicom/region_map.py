import numpy as np
import os
import pandas as pd
import re
from typing import Optional

from hnas.regions import is_region
from hnas.types import PatientID

class RegionMap:
    def __init__(
        self,
        data: pd.DataFrame):
        self.__data = data

    @staticmethod
    def load(filepath: str) -> Optional['RegionMap']:
        if os.path.exists(filepath):
            map_df = pd.read_csv(filepath, dtype={ 'patient-id': str })

            # Check that internal region names are entered correctly.
            for region in map_df.internal:
                if not is_region(region):
                    raise ValueError(f"Error in region map. '{region}' is not an internal region.")
            
            return RegionMap(map_df)
        else:
            return None

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    def to_internal(
        self,
        region: str,
        pat_id: Optional[PatientID] = None) -> str:
        # Iterate over map rows.
        for _, row in self.__data.iterrows():
            if 'patient-id' in row:
                # Skip if this map row is for a different patient.
                map_pat_id = row['patient-id']
                if isinstance(map_pat_id, str) and str(pat_id) != map_pat_id:
                    continue

            args = []
            # Add case sensitivity to regexp match args.
            if 'case-sensitive' in row:
                case_sensitive = row['case-sensitive']
                if not np.isnan(case_sensitive) and not case_sensitive:
                    args += [re.IGNORECASE]
            else:
                args += [re.IGNORECASE]
                
            # Perform match.
            if re.match(row['dataset'], region, *args):
                return row.internal
        
        return region
