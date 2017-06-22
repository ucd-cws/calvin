#!/bin/bash
#SBATCH -n 32            # Total number of processors to request (32 cores per node)
#SBATCH -p high           # Queue name hi/med/lo
#SBATCH -t 48:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=jdherman@ucdavis.edu              # address for email notification
#SBATCH --mail-type=ALL                  # email at Begin and End of job

# you can define your own variables. Access them later with dollar signs ($DIR, etc.)
DIR=mustafa-82yr
GDIR=/group/hermangrp

# IMPORTANT: Python3/Pyomo/CBC solver are all installed in group directory. Add it to the path.
export PATH=$GDIR/miniconda3/bin:$GDIR/cbc/bin:$PATH

# Export CPLEX and GUROBI solvers
export CPLEX_HOME=$GDIR/cplex1263/cplex
export GUROBI_HOME=$GDIR/gurobi652/linux64
export PATH=$PATH:$GUROBI_HOME/bin:$CPLEX_HOME/bin/x86-64_linux:$GDIR/cbc/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GUROBI_HOME/lib:$CPLEX_HOME/lib/x86-64_linux/static_pic
export GRB_LICENSE_FILE=$GDIR/gurobi652/gurobi.lic

python main-82yr.py
