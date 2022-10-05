from hnas.dataset.nifti import convert_to_training

public_datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']
dilate_regions = ['BrachialPlexus_L', 'BrachialPlexus_R', 'Cochlea_L', 'Cochlea_R', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R']

# Process datasets - do this in parallel in reality.
for dataset in public_datasets:
    # Create data for localiser.
    convert_to_training(dataset, f'{dataset}-LOC', dilate_region=dilate_regions, region='all', spacing=(4, 4, 4))

    # Create data for segmenter.
    convert_to_training(dataset, f'{dataset}-SEG', region='all', size=None, spacing=(1, 1, 2))
