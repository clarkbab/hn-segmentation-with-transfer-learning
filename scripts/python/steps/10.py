from hnas.training.segmenter import train_segmenter
from hnas.regions import RegionNames

seg_dataset = 'INST'
n_trains = [5, 10, 20, 50, 100, 200, 'all']
n_folds = 5
test_folds = [0, 1, 2, 3, 4]
n_train_epochs = {
    5: 900,             # BP_L/R @ n=5 took this long to plateau.
    10: 450,            # BP_L/R, L_L/R @ n=10.
    20: 300,            # BP_L/R, ON_L/R @ n=20.
    'default': 150      # All other models.
}

# Train segmenter network per region - in reality this would be performed across multiple machines.
for test_fold in test_folds:
    for n_train in n_trains:
        n_epochs = n_train_epochs[n_train] if n_train in n_train_epochs else n_train_epochs['default']

        for region in RegionNames:
            # Train segmenter network.
            train_segmenter(seg_dataset, region, f'segmenter-{region}', 'int-1gpu-150epochs', n_epochs=n_epochs, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
