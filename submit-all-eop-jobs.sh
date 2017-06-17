#!/bin/bash

# Example submission for SLURM system
# Loop through multiple job submissions
# Pass environment variables to job script
NUMBERS=( 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 )

for NUM in ${NUMBERS[@]}
do
  sbatch -J CAEOP${NUM} --export=EOP=${NUM} job-slurm-annual-eop.sh
done