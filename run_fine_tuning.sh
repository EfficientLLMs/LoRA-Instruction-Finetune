#!/bin/bash
#SBATCH --job-name=fine_tune_lora_70m
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/fine_tune_lora_70m.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

python instruction_tuning_raw.py \
    --wandb_name "70m_fine_tune_alpaca" \
    --base_model "EleutherAI/pythia-70m" \
    --output "./weight/pythia_70m_r=64"