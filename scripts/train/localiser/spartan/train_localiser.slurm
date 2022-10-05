#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 2-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['HN1-LOC','HNPCT-LOC','HNSCC-LOC','OPC-LOC']"
# DATASETS="PMCC-HN-TRAIN-LOC"
REGION="Parotid_L"
MODEL_NAME="localiser-$REGION"
n_EPOCHS=100
n_GPUS=1
n_NODES=1
n_WORKERS=4
PRETRAINED="('localiser-$REGION','public-1gpu-100epochs','BEST')"
RESUME=False
RESUME_CHECKPOINT="last"
RUN_NAME="tl-100epochs"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser --slurm_job_id $SLURM_JOB_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --n_epochs $n_EPOCHS --n_gpus $n_GPUS \
    --n_nodes $n_NODES --n_subset $n_SUBSET --n_workers $n_WORKERS --pretrained $PRETRAINED --region $REGION --resume $RESUME --resume_checkpoint $RESUME_CHECKPOINT \
    --use_logger $USE_LOGGER
