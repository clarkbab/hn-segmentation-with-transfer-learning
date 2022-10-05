#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

# 'DATASET', 'OUTPUT_SPACING' and 'TRAINING_DATASET' passed by 'create_jobs' script.
region="all"
DILATE_region=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
)
DILATE_region=$(join , "${DILATE_REGIONS[@]}")

command="python $HNAS_CODE/scripts/convert_to_training.py \
    --dataset $DATASET \
    --region $REGIONS \
    --dilate_region $DILATE_REGIONS \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET"
echo $command
eval $command
