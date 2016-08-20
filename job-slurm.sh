#!/bin/bash
#SBATCH -D /home/jdherman/pyvin
#SBATCH -o /home/jdherman/pyvin/job.%j.%N.out
#SBATCH -e /home/jdherman/pyvin/job.%j.%N.err
#SBATCH -n 32            # Total number of processors to request (32 cores per node)
#SBATCH -p med           # Queue name hi/med/lo
#SBATCH -t 200:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=jdherman@ucdavis.edu              # address for email notification
#SBATCH --mail-type=ALL                  # email at Begin and End of job

DIR=statewide-82yr-debug

# pyomo solve --solver=glpk --solver-suffix=dual pyvin.py $DIR/data.dat --json --report-timing --stream-solver
pyomo solve --solver=cbc --solver-suffix=dual --solver-options="threads=32" pyvin.py $DIR/data.dat --json --report-timing --stream-solver

mv results.json $DIR && cd $DIR

python ../postprocessor/postprocess.py

python ../postprocessor/debug_flow_finder.py
