from hnas.training.localiser import train_localiser
from hnas.training.segmenter import train_segmenter
from hnas.regions import RegionNames

loc_datasets = ['HN1-LOC', 'HNPCT-LOC', 'HNSCC-LOC', 'OPC-LOC']
seg_datasets = ['HN1-SEG', 'HNPCT-SEG', 'HNSCC-SEG', 'OPC-SEG']

# Train localiser/segmenter network per region - in reality this would be performed across multiple machines.
for region in RegionNames:
    # Train localiser network.
    train_localiser(loc_datasets, region, f'localiser-{region}', 'public-1gpu-150epochs', n_epochs=150)

    # Train segmenter network.
    train_segmenter(seg_datasets, region, f'segmenter-{region}', 'public-1gpu-150epochs', n_epochs=150)
