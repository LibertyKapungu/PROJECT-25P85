#!/bin/bash
#SBATCH --job-name=download_EARS
#SBATCH --output=download_EARS.out
#SBATCH --error=download_EARS.err
#SBATCH --time=02:00:00
#SBATCH --partition=jaguarcluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Download and unzip the EARS dataset
for X in $(seq -w 092 107); do
  curl -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -o p${X}.zip
  unzip p${X}.zip
  rm p${X}.zip
done