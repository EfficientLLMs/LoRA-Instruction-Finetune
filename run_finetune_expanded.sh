#!/bin/bash

combinations=(
    "1.4b 2.8b"
    "1.4b 6.9b"
    "2.8b 6.9b"
)

# Loop through each combination
for combo in "${combinations[@]}"; do
    # Split the combination into small_model and large_model
    read -r small large <<< "$combo"
    echo "Running experiment with small_model=$small and large_model=$large"
    accelerate launch instruction_tuning_expanded.py --small_model $small --large_model $large
done