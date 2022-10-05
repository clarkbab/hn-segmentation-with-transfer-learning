from hnas.dataset.nifti import convert_to_training

dataset = 'INST'

# Create data for segmenter.
convert_to_training(dataset, f'{dataset}-SEG', region='all', size=None, spacing=(1, 1, 2))
