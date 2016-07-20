To create a model run (12 months) from HOBBES network: 
```
cnf matrix --format csv --start 2002-10 --stop 2003-10 --ts . --to networklinks --max-ub 1000000 --outnodes networknodes
```

To add debug flows, use flag: 
```
--debug All
```

To run the optimization: 
```
pyomo solve --solver=glpk --report-timing --solver-suffix=dual pyvin.py data.dat --json
```

Then run `postprocess.py` to format the results as time series. 
