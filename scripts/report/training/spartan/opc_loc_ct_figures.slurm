#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:30:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

bash $SCRIPT_DIR/report/training/manual/opc_loc_ct_figures
