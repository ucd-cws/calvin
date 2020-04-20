import os
import subprocess
from calvin import *

# directory for links, inflows, sr_dict, ...and result output
model_dir = './my-models/two-sr-pilot-lf'
flow_column = 'hist'

# particle swarm search settings and
pso_options = {'c1': 2.2, 'c2': 1.2, 'w': 0.7}
pso_velocity_clamp = [-100., 100.]
pso_particle_count = 5
# COSVF 
lower_bounds = [-250., -2500., -250., -3000., -250., -1500., -100., -500.]

# Pyro4 server solver node count
server_solver_nodes = 5
# start the pyro server (will recognize if already started)
subprocess.call('./start_pyro.sh {}'.format(server_solver_nodes), shell=True)

# conduct pso for cosvf:
cosvfpso = COSVF_PSO(
	solver='gurobi',
    links_dir=model_dir,
    flow_column=flow_column,
    n_particles=pso_particle_count,
    options=pso_options,
    velocity_clamp=pso_velocity_clamp,
    lower_bounds=lower_bounds)

# optimize Swarm
cosvfpso.optimize(
    iters=2
)
