## Installation

Requires: [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [Pandas](http://pandas.pydata.org/), [Pyomo](https://software.sandia.gov/downloads/pub/pyomo/PyomoInstallGuide.html), and [pwlf](https://pypi.org/project/pwlf/).

Recommended command-line method to install Pyomo (**only tested with Python 3.4**):
```bash 
conda install pyomo pyomo.extras glpk --channel cachemeorg
```
This will install the [GLPK](https://www.gnu.org/software/glpk/) solver. Pyomo can also connect to other solvers, including [CBC](https://projects.coin-or.org/Cbc) and [CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/), and [Gurobi](http://www.gurobi.com/). Installation of these solvers is not covered here. *UC Davis: these are installed on HPC1 in `/group/hermangrp/`*.
