#!/bin/bash
#SBATCH --job-name=baseline_noisy
#SBATCH --output=baseline_noisy.out
#SBATCH --error=baseline_noisy.err
#SBATCH --time=10:00:00
#SBATCH --partition=jaguarcluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Activate your environment
source ~/.bashrc
conda activate py312

# Run your Python script
python /home/25p85/Gabi/PROJECT-25P85/src/experiments/EXP3/EXP3_SS_param_sweep_SS_and_GTCRN.py