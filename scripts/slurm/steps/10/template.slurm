#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
##SBATCH --time 5-0:00:00
#SBATCH --time 0-0:30:00
## Will need to edit the following 'SBATCH' lines:
#SBATCH --account punim1413
#SBATCH --partition feit-gpu-a100
#SBATCH --qos feit

source ~/venvs/transfer-learning/bin/activate

version=$(python --version)
echo $version

# 'DATASET', 'N_EPOCHS', 'N_TRAIN' and 'TEST_FOLD' passed by 'create_jobs' script.
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-${REGION}"
N_FOLDS=5
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CHECKPOINT=None
RUN_NAME="inst-1gpu-150epochs"
USE_LOGGER=False        # Set 'True' to use 'wandb' logging. Must set this up first (https://wandb.ai).

# Check if 'N_TRAIN' training samples exist for this region.
N_TRAIN_MAX = $(python $HNAS_CODE/scripts/train/segmenter/get_n_train_max.py \
    --dataset $DATASETS \
    --n_folds $N_FOLDS \
    --region $REGION \
    --test_fold $TEST_FOLD)

if [ \( "$N_TRAIN" == "all" \) -o \( "$N_TRAIN" -ge "$N_TRAIN_MAX" \) ]; then
    echo "N_TRAIN ($N_TRAIN) >= N_TRAIN_MAX ($N_TRAIN_MAX). Skipping."
    exit 0
fi

command="python $HNAS_CODE/scripts/train/segmenter/train_segmenter.py \
    --dataset $DATASETS \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --region $REGION \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER"
echo $command
eval $command
