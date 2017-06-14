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

python main.py

# if you want GLPK (serial)
#pyomo solve --solver=glpk --solver-suffix=dual pyvin.py $DIR/data.dat --json --report-timing --stream-solver

# if you want CBC (parallel, can only use up to 32 cores)
# if you want to use CPLEX or GUROBI solvers, change "--solver=cbc" to "--solver=cplex" or "--solver=gurobi"
# pyomo solve --solver=cbc --solver-suffix=dual --solver-options="threads=32" pyvin.py $DIR/data.dat --json --report-timing --stream-solver

# the --stream-solver flag gives verbose output if you want to see what's going on. If you don't care about this, remove it.

# At this point the results.json file should exist. Now we can postprocess. (This does not run in parallel; you could do it separately after the job finishes)

# mv results.json $DIR && cd $DIR

# python ../postprocessor/postprocess.py

# python ../postprocessor/debug_flow_finder.py