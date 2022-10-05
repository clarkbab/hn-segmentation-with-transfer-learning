#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 7-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['HN1-LOC','HNPCT-LOC','HNSCC-LOC','OPC-LOC']"
region=(
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
n_EPOCHS=150
N_FOLDS=None
n_GPUS=1
n_NODES=1
n_WORKERS=4
RESUME=True
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True
TEST_FOLD=None
RUN_NAME="public-1gpu-150epochs"

python $SCRIPT_DIR/train/localiser --slurm_array_job_id $SLURM_ARRAY_JOB_ID --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --n_epochs $n_EPOCHS \
    --n_folds $N_FOLDS --n_gpus $n_GPUS --n_nodes $n_NODES --n_workers $n_WORKERS --region $REGION --resume $RESUME --resume_checkpoint $RESUME_CHECKPOINT \
    --test_fold $TEST_FOLD --use_logger $USE_LOGGER
