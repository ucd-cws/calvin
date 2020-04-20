# conducts perfect forsight run

import os
from calvin import *

model_dir = './my-models/calvin-pf/'

# create a CALVIN links instance
calvin = CALVIN(os.path.join(model_dir,'links82yr.csv')

# create a Pyomo network model from links
calvin.create_pyomo_model(debug_mode=False)

# solve the Pyomo network flow model
calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=False)
postprocess(calvin.df, calvin.model, 
	resultdir=os.path.join(model_dir,'results'))

# final objective value
print(value(calvin.model.total()))
