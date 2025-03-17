#!/bin/bash

apikey=$1
folder=$2

# Upload results to the server
# python3.9 upload_results.py -m upload -a 9cfaf2bb33bd321a7730903a45ea0a45 -i /Users/smukherjee/Downloads/analyzing_automl_results/v3/results/flaml.openml_t_3623.test.singularity.20250312T182154
for subfolder in $folder/*; do
    python3.9 ./automlbenchmark/upload_results.py -m check -a $apikey -i $subfolder
done