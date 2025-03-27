#!/bin/bash

# Define variables
RESULTS_DIR=$1 # Replace with actual path
AUTOMLB_DIR=$2
API_KEY=$3

# Loop through result files and upload them
for result in "$RESULTS_DIR"/*; do
  python "$AUTOMLB_DIR/upload_results.py" -m upload -a "$API_KEY" -i "$result"
done
