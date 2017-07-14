from calvin import *

eop = None

for i in range(1922,2004):

  print('\nNow running WY %d' % i)

  calvin = CALVIN('calvin/data/annual/linksWY%d.csv' % i, ic=eop)

  calvin.eop_constraint_multiplier(0.1)

  calvin.create_pyomo_model(debug_mode=True, debug_cost=2e8)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=True, maxiter=15)

  calvin.create_pyomo_model(debug_mode=False)
  calvin.solve_pyomo_model(solver='glpk', nproc=1, debug_mode=False)

  # this will append to results files
  eop = postprocess(calvin.df, calvin.model, resultdir='results-annual', annual=True) 
