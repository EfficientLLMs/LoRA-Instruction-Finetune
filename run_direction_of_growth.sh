#!/bin/bash
#SBATCH --job-name=direction_of_growth
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm_logs/direction_of_growth.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

nvidia-smi

python src/direction_of_growth.py