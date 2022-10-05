#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

# 'DATASET' passed by 'create_jobs' script.
REGIONS="all"
ANONYMISE=False     # Use ANONYMISE=True if moving data offsite for training.
REGION=$(join , "${REGIONS[@]}")

command="python $HNAS_CODE/scripts/process/dicom/convert_to_nifti.py \
    --dataset $DATASET \
    --regions $REGION \
    --anonymise $ANONYMISE"
echo $command
eval $command
