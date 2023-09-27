#!/bin/bash

#SBATCH --job-name=test_slurm_arrays
#SBATCH --array=1-5
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --account=laionize
#SBATCH --partition=booster
#SBATCH --output=slurm_logs/slurm-%A_%a.out

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
srun python -c "$(cat experiments_list | sed -n -e "${SLURM_ARRAY_TASK_ID}p")"