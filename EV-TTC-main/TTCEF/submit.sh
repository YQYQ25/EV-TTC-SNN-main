#!/bin/bash
#SBATCH --job-name=ttcef_data_gen
#SBATCH --array=0-34
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=array_%A-%a.out
#SBATCH  --cpus-per-task=18
# Print the task id.
cd /path-to-repo/TTCEF
python create_exp.py --num $SLURM_ARRAY_TASK_ID --data_dir /tmp/
python calc_gt.py --num $SLURM_ARRAY_TASK_ID --data_dir /tmp/