## PyVIN Network Flow Optimization Model

### PyVIN is still in active development and is not production-ready

Network flow optimization of California's water supply system. Requires: [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)/[Pandas](http://pandas.pydata.org/) (all available in the [Anaconda Distribution]((https://www.continuum.io/downloads)), and [Pyomo](https://software.sandia.gov/downloads/pub/pyomo/PyomoInstallGuide.html).

Recommended command-line method to install Pyomo (**only tested with Python 3.4**):
```bash 
conda install pyomo pyomo.extras glpk --channel cachemeorg
```
This will install the [GLPK](https://www.gnu.org/software/glpk/) solver. Pyomo can also connect to other solvers, including [CBC](https://projects.coin-or.org/Cbc) and [CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/), and [Gurobi](http://www.gurobi.com/). Installation of these solvers is not covered here. *UC Davis: these are installed on HPC1 in `/group/hermangrp/`*.



## Quick Start

- Clone repository
```
git clone https://github.com/msdogan/pyvin
```

- Get network data (links). CSV file with column headers:
```
i,j,k,cost,amplitude,lower_bound,upper_bound
```
Where `i,j,k` are the source node, destination node, and piecewise index for the network problem. Each row is a link. The California network data files are too large to host on Github; they can be downloaded here:
  + [1-year example (WY 1922, 1 CSV file, 400 KB)](https://www.dropbox.com/s/9aq7aaom4dvn0b5/linksWY1922.csv.zip?dl=1)
  + [82-year perfect foresight (1 CSV file, 27 MB)](https://www.dropbox.com/s/ikt5j6kd7n80rir/links82yr.csv.zip?dl=1)
  + [Annual, limited foresight (ZIP of 82 CSV files, 31 MB)](https://www.dropbox.com/s/ac1gxs8y49oiw7d/annual.zip?dl=1)

  To export other subsets of the network (in space or time), see the [advanced readme](calvin/data) for data export from [HOBBES](https://hobbes.ucdavis.edu/node).

- Create a Python script to import the network data and run the optimization. It is recommended to first run in "debug mode" to identify and remove infeasibilities in the network.
  ```python
  # main-example.py
  from calvin import *

  calvin = CALVIN('linksWY1922.csv')

  # run in debug mode. reduces LB constraints.
  calvin.create_pyomo_model(debug_mode=True, debug_cost=2e10)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=True)

  # run without debug mode (should be feasible)
  calvin.create_pyomo_model(debug_mode=False)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=False)

  postprocess(calvin.df, calvin.model, resultdir='example-results')
  # creates output CSV files in the directory specified
  ```

  Running `python main-example.py` on the command line will show:
  ```bash
  Creating Pyomo Model (debug=True)
  -----Solving Pyomo Model (debug=True)
  Finished. Fixing debug flows...
  SR_ML.1922-09-30_FINAL UB raised by 8.28 (0.28%)
  -----Solving Pyomo Model (debug=True)
  Finished. Fixing debug flows...
  All debug flows eliminated (iter=2, vol=8.28)
  Creating Pyomo Model (debug=False)
  -----Solving Pyomo Model (debug=False)
  Optimal Solution Found (debug=False).
  ```

- The folder `example-results` will contain 8 CSV files. All are timeseries data, each row is 1 month. Recommended to read these into `pandas` for further analysis: `df = pd.read_csv(filename, index_col=0, parse_dates=True)`. 
  + `flow.csv` (flows on links, TAF/month, columns are link names)
  + `storage.csv` (end-of-month surface and GW storage, TAF)
  + `dual_lower.csv` (dual values on lower bound constraints)
  + `dual_upper.csv` (dual values on upper bound constraints)
  + `dual_node.csv` (dual values on mass balance constraints)
  + `shortage_volume.csv` (water supply shortage, relative to demand, on selected links and aggregated regions)
  + `shortage_cost.csv` (cost of water supply shortage, for selected links and aggregated regions)
  + `evaporation.csv` (TAF/month)


## Running in parallel

Several of the solvers available through Pyomo support shared-memory parallelization. (Importantly GLPK is one exception that does not support parallelization). To take advantage of this, change the script above to include:
```python
calvin.solve_pyomo_model(solver='cplex', nproc=32, debug_mode=True)
# do the same again for the non-debug mode run
```

Several job scripts are included to support running on a SLURM cluster such as [HPC1](http://ssg.cs.ucdavis.edu/services/research/hpc1-cluster) at UC Davis. These will need to be customized for each system. 


## Example Data Visualization: Supply Portfoilio

In general, plotting results is left to the user. A few useful plot types will be included in `calvin/plots.py`. One example is the supply portfolio stacked bar chart, which plots the sum of flows by each region, supply type, and urban/agricultural link type:

![PyVIN Supply Portfolio Figure](documentation/supply_portfolio.png)

## More Info
The [Documentation](documentation/pyvin_documentation.pdf) describes the model in more detail. This refers to an earlier version of the model using Pyomo's `AbstractModel` type, but the setup is mostly the same in the current `ConcreteModel`. There is also detailed [Pyomo documentation](http://www.pyomo.org/documentation/).
  
