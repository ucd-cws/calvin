Follow these 3 steps to install HOBBES network format components:

Step 1: GitHub installation
https://docs.google.com/document/d/1b9SrHkWDmSJbUfp9oWGVoc9iDkROO3CiRScsc5OGYD4/edit

Step 2: Nodejs installation
https://docs.google.com/document/d/1ExARl8gffVxaG3zs-FJP-zBJejKk_KNxuJPYjFg_Vqk/edit#heading=h.53mxy8k62q2v

Step 3: Install and run network code
https://docs.google.com/document/d/1lGeftVEqG29oMpNMte_GTC2krhkZutpp-2GdsonTzgA/edit#heading=h.ynb4aay64kqp


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
