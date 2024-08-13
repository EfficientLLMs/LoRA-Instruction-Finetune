#!/bin/bash
#SBATCH --job-name=fine_tune_lora_70m_410m_1e-4_noop
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/fine_tune_lora_70m_410m_1e-4_noop.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

python instruction_tuning_expanded.py --lr 1e-4 --large_adapter "./weight/pythia_70m_lora_expanded_noop_r=64" --output "./weight/pythia_410m_lora_expanded_noop_r=64/" --scheduler