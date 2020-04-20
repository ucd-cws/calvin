import os
from calvin import *

# directory for links, inflows, sr_dict, ...and result output
model_dir = './my-models/two-sr-pilot-lf'

# flow column of inflows.csv to load
flow_column = 'hist'

# load inflows
inflows = load_lf_inflows(model_dir,flow_column)

# load dictionary of reservoirs
sr_dict = load_sr_dict(model_dir)

### COSVF params ###
# the list of COSVF params [Pmin, Pmax, ...] must match the sr_dict order
pmin_pmax = [-52.05, -1680.88, -52.05, -1935.86, -52.05, -993.69, -2.4, -2.9]

calvin = CALVIN(
  linksfile=os.path.join(model_dir,'links.csv'),
  cosvf_pminmax=pmin_pmax,
  sr_dict=sr_dict,
  inflows=inflows
)

calvin.cosvf_solve(
  results_dir=os.path.join(model_dir,'results',flow_column),
  solver='gurobi'
)
