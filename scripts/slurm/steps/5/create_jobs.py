import subprocess

script = 'scripts/slurm/steps/5/template.slurm'
datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']
names = ['LOC', 'SEG']
spacings = ['\'(4,4,4)\'', '\'(1,1,2)\'']

for dataset in datasets:
    for name, spacing in zip(names, spacings):
        # Create slurm command.
        training_dataset = f'{dataset}-{name}'
        export = f'ALL,DATASET={dataset},OUTPUT_SPACING={spacing},TRAINING_DATASET={training_dataset}'
        command = f'sbatch --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
