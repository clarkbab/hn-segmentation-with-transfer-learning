from hnas.dataset.dicom import convert_to_nifti

public_datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']

# Process datasets - do this in parallel in reality.
for dataset in public_datasets:
    convert_to_nifti(dataset, region='all', anonymise=False)
