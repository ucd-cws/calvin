# PyVIN Network Flow Optimization Model

Follow these 3 steps to install HOBBES network format components:

Step 1: GitHub installation

https://docs.google.com/document/d/1b9SrHkWDmSJbUfp9oWGVoc9iDkROO3CiRScsc5OGYD4/edit

Step 2: Nodejs installation

https://docs.google.com/document/d/1ExARl8gffVxaG3zs-FJP-zBJejKk_KNxuJPYjFg_Vqk/edit#heading=h.53mxy8k62q2v

Step 3: Install and run network code

https://docs.google.com/document/d/1lGeftVEqG29oMpNMte_GTC2krhkZutpp-2GdsonTzgA/edit#heading=h.ynb4aay64kqp


To create a model run (12 months) from HOBBES network: 
```
cnf matrix --format tsv --start 2002-10 --stop 2003-10 --ts . --fs :tab: --to networklinks --max-ub 1000000 --outnodes networknodes --verbose
```

To add debug flows, use flag: 
```
--debug All
```

To create a run with defined nodes only:

Example: SR_SHA and D5 between Oct 1983 to Sep 1984 in debug mode
```
cnf matrix --format tsv --start 1983-10 --stop 1984-10 --ts . --fs :tab: --to networklinks --max-ub 10000000 --outnodes networknodes nodes SR_SHA D5 --debug SR_SHA,D5 --verbose
```

To run the optimization: 
```
pyomo solve --solver=glpk --solver-suffix=dual pyvin.py data.dat --json --report-timing --stream-solver
```

Then run `postprocess.py` to format the results as time-series. 
