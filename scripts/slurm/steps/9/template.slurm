#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

# 'DATASET', 'OUTPUT_SPACING' and 'TRAINING_DATASET' passed by 'create_jobs' script.
REGIONS="all"

command="python $HNAS_CODE/scripts/process/nifti/convert_to_training.py \
    --dataset $DATASET \
    --region $REGIONS \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET"
echo $command
eval $command
