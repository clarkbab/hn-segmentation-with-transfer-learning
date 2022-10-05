import subprocess

script = 'scripts/slurm/steps/9/template.slurm'
dataset = 'INST'
spacing = '"(1,1,2)"'
training_dataset = f'{dataset}-SEG'

# Create slurm command.
export = f'ALL,DATASET={dataset},OUTPUT_SPACING={spacing},TRAINING_DATASET={training_dataset}'
command = f'sbatch --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
