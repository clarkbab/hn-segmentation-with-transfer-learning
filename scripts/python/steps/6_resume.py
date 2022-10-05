from hnas.training.localiser import train_localiser
from hnas.training.segmenter import train_segmenter
from hnas.training import train

models = ['localiser', 'segmenter']
datasets = {
    'localiser': ['HN1-LOC', 'HNPCT-LOC', 'HNSCC-LOC', 'OPC-LOC'],
    'segmenter': ['HN1-SEG', 'HNPCT-SEG', 'HNSCC-SEG', 'OPC-SEG']
}
region = 'Brain'

# For failed localiser.
train_localiser(datasets, region, f'localiser-{region}', 'public-1gpu-150epochs', n_epochs=150, resume=True, resume_checkpoint='last')

# For failed segmenter.
train_segmenter(datasets, region, f'segmenter-{region}', 'public-1gpu-150epochs', n_epochs=150, resume=True, resume_checkpoint='last')
