#!/bin/bash

# Example submission for SLURM system
# Loop through multiple job submissions
# Pass environment variables to job script
NUMBERS=( 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 )

for NUM in ${NUMBERS[@]}
do
  sbatch -J CAEOP${NUM} --export=EOP=${NUM} job-slurm-annual.sh
done