#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="clustering.py"

# Define the values of kmeans you want to iterate over
VALUES=( {3..10} )

OUTPUT_FILE="./logs/clustering_runs.txt"

# Redirect all output to the output file (overwrite mode)
> "$OUTPUT_FILE"

# Iterate over each value
for k in "${VALUES[@]}"
do
    echo "Running with kmeans=$k" | tee -a "$OUTPUT_FILE"
    python3 "$PYTHON_SCRIPT" -k "$k" >> "$OUTPUT_FILE" 2>&1
    echo "-------------------------------------------------" | tee -a "$OUTPUT_FILE"
done