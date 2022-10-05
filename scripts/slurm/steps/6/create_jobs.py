import subprocess

regions = '0-16'
regions = '0'
models = ['localiser', 'segmenter']
model_abbrs = ['LOC', 'SEG']

for model, model_abbr in zip(models, model_abbrs):
    script = f'scripts/slurm/steps/6/{model}_template.slurm'
    dataset = f"\"['HN1-{model_abbr}','HNPCT-{model_abbr}','HNSCC-{model_abbr}','OPC-{model_abbr}']\""

    # Create slurm command.
    export = f'ALL,DATASET={dataset}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
