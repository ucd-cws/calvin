from pyomo.environ import *
import pandas as pd
from calvin import *

calvin = Network('calvin/data/links1yr.csv')
calvin.create_pyomo_model()
calvin.solve_pyomo_model(solver='glpk', nproc=1, verbose=False)
# calvin.results.write(filename='thing.json', format='JSON')
postprocess(calvin.df, calvin.model)
