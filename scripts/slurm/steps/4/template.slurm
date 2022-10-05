#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-6:00:00

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

REGION='all'
ANONYMISE=False

command="python $HNAS_CODE/scripts/convert_to_nifti.py \
    --dataset $DATASET \
    --region $REGION \
    --anonymise $ANONYMISE"
echo $command
eval $command
