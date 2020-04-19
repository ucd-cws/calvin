## Quick Start

- Clone repository
```
git clone https://github.com/ucd-cws/calvin
```

- Get network data (links). CSV file with column headers:
```
i,j,k,cost,amplitude,lower_bound,upper_bound
```
Where `i,j,k` are the source node, destination node, and piecewise index for the network problem. Each row is a link. The California network data files are too large to host on Github; they can be downloaded here:
  + [1-year example (WY 1922, 1 CSV file, 400 KB)](https://www.dropbox.com/s/9aq7aaom4dvn0b5/linksWY1922.csv.zip)
  + [82-year perfect foresight (1 CSV file, 27 MB)](https://www.dropbox.com/s/ikt5j6kd7n80rir/links82yr.csv.zip)
  + [Annual, limited foresight (ZIP of 82 CSV files, 31 MB)](https://www.dropbox.com/s/ac1gxs8y49oiw7d/annual.zip)

  To export other subsets of the network (in space or time), see the [advanced readme](../data/california-network) for data export from [HOBBES](https://hobbes.ucdavis.edu/node).

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
