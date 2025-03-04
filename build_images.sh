#!/bin/bash

# User to use
usertouse=$1
mode=$2

# Load required modules
module load 2022
# module spider Anaconda3/2022.05
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
module load Python/3.10.4-GCCcore-11.3.0

# Create conda environment only if it doesn't exist
if ! conda info --envs | grep -q "automl"; then
  conda create -n automl python=3.9.19 -y
fi

# Activate environment
yes | conda activate /home/$usertouse/.conda/envs/automl

# Install required Python packages
pip install --user -r /home/$usertouse/OpenML-auto-automl/requirements.txt

# Define list of frameworks
frameworks=("autosklearn" "flaml" "gama" "h2oautoml")
# frameworks=("h2oautoml")

cd /home/$usertouse/automlbenchmark/

# Run benchmarks
for framework in "${frameworks[@]}"; do
  echo $framework
  yes | python runbenchmark.py "$framework" openml/t/3812 --mode "$mode" --setup only -o /home/$usertouse/automl_data/results
done

