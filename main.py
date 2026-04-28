from calvin import *

# specify data file
calvin = CALVIN('network_1921_1922_debug.csv',log_name="calvin")

calvin.no_gw_overdraft() # no gw overdraft

# calvin.inflow_multiplier(0.9) # reduce all inflows by 10% 
# calvin.inflow_multiplier(1.1) # increase all inflows by 10% 

# create pyomo model from specified data file
calvin.create_pyomo_model(save_final_csv=False)

# solve the problem
calvin.solve_pyomo_model(solver='glpk', nproc=1, tee=True)

resultdir = 'results_1921_1922_no_overdraft_debug'

# postprocess results to create time-series files
postprocess(calvin.df, calvin.model, resultdir=resultdir)

aggregate_regions(resultdir) # aggregate and append regionwide results and plot
