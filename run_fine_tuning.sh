#!/bin/bash
#SBATCH --job-name=fine_tune_lora_70m_410m_1e-5
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/fine_tune_lora_70m_410m_1e-5.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

python instruction_tuning_expanded.py --lr 1e-5