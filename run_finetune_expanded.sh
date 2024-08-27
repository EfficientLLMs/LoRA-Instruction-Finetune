#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/fine_tune_expanded_small_schedule.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

# combinations=(
#     "1.4b 2.8b"
#     "1.4b 6.9b"
#     "2.8b 6.9b"
# )

combinations=(
    "70m 160m"
    "70m 410m"
    "160m 410m"
)

# Loop through each combination
for combo in "${combinations[@]}"; do
    # Split the combination into small_model and large_model
    read -r small large <<< "$combo"
    echo "Running experiment with small_model=$small and large_model=$large"
    accelerate launch instruction_tuning_expanded.py --small_model $small --large_model $large --expand_method padding --scheduler
done