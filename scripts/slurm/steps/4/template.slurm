#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-6:00:00

echo $HNAS_CODE
echo $HNAS_DATA

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

REGION='all'
ANONYMISE=False

command="python $HNAS_CODE/scripts/slurm/steps/4/convert_to_nifti.py \
    --dataset $DATASET \
    --region $REGION \
    --anonymise $ANONYMISE"
echo $command
eval $command
