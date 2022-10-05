import subprocess

script = 'scripts/slurm/steps/8/template.slurm'
dataset = 'INST'

# Create slurm command.
export = f'ALL,DATASET={dataset}'
command = f'sbatch --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
