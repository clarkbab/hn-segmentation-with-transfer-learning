import subprocess

# Example to resume training 'localiser-Brain' model.
model = 'localiser'     # model = 'localiser' or 'segmenter'.
# model = 'segmenter'
region = '0'            # region = '0' to '16' or range e.g. '0-5'.
if model == 'localiser':
    model_abbr = 'LOC'
elif model == 'segmenter':
    model_abbr = 'SEG'
script = f'scripts/slurm/steps/6/resume_{model}_template.slurm'
dataset = f"\"['HN1-{model_abbr}','HNPCT-{model_abbr}','HNSCC-{model_abbr}','OPC-{model_abbr}']\""

# Create slurm command.
export = f'ALL,DATASET={dataset}'
command = f'sbatch --array={region} --export={export} {script}' 
print(command)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
process.communicate()
