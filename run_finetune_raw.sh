#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/fine_tune_raw_small.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


# Define an array of model names
names=("70m" "160m" "410m")

# Loop through each name
for name in "${names[@]}"
do
    echo "Running with --name $name"
    accelerate launch instruction_tuning_raw.py --name $name --scheduler
done
