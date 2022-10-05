import collections
import pandas as pd
import pydicom as dcm
from typing import Dict, List, Optional, OrderedDict

from hnas import logging
from hnas.types import PatientRegion, PatientRegions

from .ct_series import CTSeries
from .dicom_file import DICOMFile, SOPInstanceUID
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter

class RTSTRUCT(DICOMFile):
    def __init__(
        self,
        series: 'RTSTRUCTSeries',
        id: SOPInstanceUID,
        region_dups: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None):
        self.__global_id = f"{series} - {id}"
        self.__id = id
        self.__ref_ct = None        # Lazy-loaded.
        self.__region_dups = region_dups
        self.__region_map = region_map
        self.__series = series

        # Get index.
        index = self.__series.index
        index = index[index['sop-id'] == self.__id]
        self.__index = index
        self.__check_index()
        self.__path = self.__index.iloc[0]['filepath']

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def path(self) -> str:
        return self.__path

    @property
    def ref_ct(self) -> str:
        if self.__ref_ct is None:
            self.__load_ref_ct()
        return self.__ref_ct

    @property
    def series(self) -> str:
        return self.__series

    def get_rtstruct(self) -> dcm.dataset.FileDataset:
        return dcm.read_file(self.__path)

    def get_region_info(
        self,
        use_mapping: bool = True) -> Dict[int, Dict[str, str]]:
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region IDs.
        roi_info = RTSTRUCTConverter.get_roi_info(rtstruct)

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        roi_info = dict(filter(lambda i: RTSTRUCTConverter.has_roi_data(rtstruct, i[1]['name']), roi_info.items()))

        # Map to internal names.
        if use_mapping and self.__region_map:
            pat_id = self.__series.study.patient.id
            def map_name(info):
                info['name'] = self.__region_map.to_internal(info['name'], pat_id=pat_id)
                return info
            roi_info = dict((id, map_name(info)) for id, info in roi_info.items())

        return roi_info

    def has_region(
        self,
        region: PatientRegion,
        use_mapping: bool = True) -> bool:
        return region in self.list_regions(use_mapping=use_mapping)

    def list_regions(
        self,
        use_mapping: bool = True) -> List[PatientRegion]:
        # Get region names.
        rtstruct = self.get_rtstruct()
        regions = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        regions = list(filter(lambda r: RTSTRUCTConverter.has_roi_data(rtstruct, r), regions))

        # Map to internal regions.
        if use_mapping and self.__region_map is not None:
            pat_id = self.__series.study.patient.id
            new_regions = []
            for region in regions:
                mapped_region = self.__region_map.to_internal(region, pat_id=pat_id)
                # Don't map regions that would map to an existing region name.
                if mapped_region != region and mapped_region in regions:
                    logging.warning(f"Mapped region '{mapped_region}' (mapped from '{region}') already found in unmapped regions for '{self}'. Skipping...")
                    new_regions.append(region)
                else:
                    new_regions.append(mapped_region)
            regions = new_regions

        # Check for regions with the same name.
        # This can occur in the RTSTRUCT dicom. E.g. 'HNSCC' dataset, patient 'HNSCC-01-0202', region 'final iso'.
        dup_regions = []
        for region in regions:
            if region in dup_regions and not (self.__region_dups is not None and region in self.__region_dups):
                raise ValueError(f"Duplicate region '{region}' found for RTSTRUCT '{self}'. Perhaps 'region-map.csv' mapped two different regions to the same name?")
            dup_regions.append(region)

        return regions

    def region_data(
        self,
        region: PatientRegions = 'all',
        use_mapping: bool = True) -> OrderedDict:
        self.__assert_requested_region(region, use_mapping=use_mapping)

        # Get region names - include unmapped as we need these to load RTSTRUCT regions later.
        unmapped_names = self.list_regions(use_mapping=False)
        names = self.list_regions(use_mapping=use_mapping)
        names = list(zip(names, unmapped_names))

        # Filter on requested regions.
        def fn(pair):
            name, _ = pair
            if type(region) == str:
                if region == 'all':
                    return True
                else:
                    return name == region
            else:
                return name in region
        names = list(filter(fn, names))

        # Get reference CTs.
        cts = self.ref_ct.get_cts()

        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Add ROI data.
        region_dict = {}
        for name, unmapped_name in names:
            data = RTSTRUCTConverter.get_roi_data(rtstruct, unmapped_name, cts)
            region_dict[name] = data

        # Create ordered dict.
        ordered_dict = collections.OrderedDict((n, region_dict[n]) for n in sorted(region_dict.keys())) 

        return ordered_dict

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTSTRUCT '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTSTRUCTs found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __assert_requested_region(
        self,
        region: PatientRegions = 'all',
        use_mapping: bool = True) -> None:
        if type(region) == str:
            if region != 'all' and not self.has_region(region, use_mapping=use_mapping):
                raise ValueError(f"Requested region '{region}' not present for RTSTRUCT '{self}'.")
        elif hasattr(region, '__iter__'):
            for r in region:
                if not self.has_region(r, use_mapping=use_mapping):
                    raise ValueError(f"Requested region '{r}' not present for RTSTRUCT '{self}'.")
        else:
            raise ValueError(f"Requested regions '{region}' isn't 'str' or 'iterable'.")

    def __load_ref_ct(self) -> None:
        rtstruct = self.get_rtstruct()
        ct_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
        self.__ref_ct = CTSeries(self.__series.study, ct_id)

    def __str__(self) -> str:
        return self.__global_id
