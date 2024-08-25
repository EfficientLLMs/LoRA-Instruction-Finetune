#!/bin/bash

# Define an array of model names
names=("1.4b" "2.8b" "6.9b")

# Loop through each name
for name in "${names[@]}"
do
    echo "Running with --name $name"
    accelerate launch instruction_tuning_raw.py --name $name
done
