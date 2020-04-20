# conducts annual limited foresight run with fixed multipliers

import os
from calvin import *

model_dir = './my-models/calvin-annual/'
eop_multiplier = 0.1

# start with blank eop
eop = None

# loop through years in period of analysis
for i in range(1922,1923+1):

  print('\nNow running WY %d' % i)

  calvin = CALVIN(os.path.join(model_dir,'annual/linksWY%d.csv' % i), ic=eop)

  calvin.eop_constraint_multiplier(eop_multiplier)

  calvin.create_pyomo_model(debug_mode=True, debug_cost=2e8)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=True, maxiter=15)

  calvin.create_pyomo_model(debug_mode=False)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=False)

  # this will append to results files
  eop = postprocess(calvin.df, calvin.model, 
  	resultdir=os.path.join(model_dir,'results'), annual=True) 
