import subprocess

script = 'scripts/slurm/steps/4/template.slurm'
datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']

for dataset in datasets:
    # Create slurm command.
    export = f'ALL,DATASET={dataset}'
    command = f'sbatch --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
