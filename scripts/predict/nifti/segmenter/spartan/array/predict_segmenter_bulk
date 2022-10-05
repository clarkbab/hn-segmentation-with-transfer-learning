import subprocess

regions = '7'
script = 'scripts/predict/nifti/segmenter/spartan/array/predict_segmenter'
test_folds = [0]
models = ['public', 'clinical', 'transfer']
n_trains = [5, 10, 20, 50, 100, 200, None]

for model in models:
    for test_fold in test_folds:
        if model == 'public':
            # Create slurm command.
            export = f'ALL,MODEL={model},TEST_FOLD={test_fold}'
            command = f'sbatch --array={regions} --export={export} {script}' 
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
        else:
            for n_train in n_trains:
                # Create slurm command.
                export = f'ALL,MODEL={model},N_TRAIN={n_train},TEST_FOLD={test_fold}'
                command = f'sbatch --array={regions} --export={export} {script}' 
                print(command)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.communicate()
