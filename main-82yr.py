from pyomo.environ import *
import pandas as pd
from calvin import *

calvin = CALVIN('calvin/data/links82yr.csv')
# calvin.inflow_multiplier(0.9)

# run in debug mode. reduces LB constraints.
# this could be an infinite loop if it never gets rid of debug flows.
calvin.create_pyomo_model(debug_mode=True, debug_cost=2e10)
# calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=True)
calvin.solve_pyomo_model(solver='gurobi', nproc=32, debug_mode=True)

# run without debug mode (should be feasible)
calvin.create_pyomo_model(debug_mode=False)
# calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=False)
calvin.solve_pyomo_model(solver='gurobi', nproc=32, debug_mode=False)

# optional: write to json file. better to postprocess and save as csv.
# calvin.results.write(filename='thing.json', format='JSON')
postprocess(calvin.df, calvin.model)
