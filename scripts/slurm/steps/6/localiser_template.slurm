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

# 'DATASET' passed by 'create_jobs' script.
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
MODEL_NAME="localiser-${REGION}"
N_EPOCHS=150
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CHECKPOINT=None
USE_LOGGER=True        # Set 'True' to use 'wandb' logging. Must set this up first (https://wandb.ai).
RUN_NAME="public-1gpu-150epochs"

command="python $HNAS_CODE/scripts/train/localiser/train_localiser.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --region $REGION \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --use_logger $USE_LOGGER"
echo $command
eval $command
