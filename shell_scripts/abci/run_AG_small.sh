#!/bin/sh

#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o /groups/gaa50073/atom/soccertrack-v2/outputs/abci_logs/run_AG_small/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.10
source ~/.bash_profile

source /groups/gaa50073/atom/soccertrack-v2/.venv/bin/activate

echo "Running command $@"
$@