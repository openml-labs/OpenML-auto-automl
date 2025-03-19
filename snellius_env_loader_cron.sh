usertouse="smukherjee"
module load 2022
module spider Anaconda3/2022.05
source /sw/arch/RHEL8/EB_production/2022/software/Anaconda3/2022.05/etc/profile.d/conda.sh
module load Python/3.10.4-GCCcore-11.3.0
yes | conda activate /home/$usertouse/.conda/envs/automl
