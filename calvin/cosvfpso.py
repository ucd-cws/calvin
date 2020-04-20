import os, shutil, json, itertools, csv, logging
from pyutilib.services import TempfileManager
import tempfile
from collections import namedtuple
from attr import attrib, attrs
from attr.validators import instance_of
from scipy.spatial import cKDTree
from .calvin import *
from .postprocessor import *
from .utils import *


class COSVF_PSO():


    def __init__(self, solver, links_dir, flow_column, n_particles, options, lower_bounds,
                 upper_bounds=None, velocity_clamp=None):
        """
        Initialize the swarm

        The COSVF-PSO swarm is initialized with a user-specified number of particles. The initial
        positions of each particle are generated randomly between the user-specified
        lower bounds and upper bounds.

        A seperate CALVIN-pyomo model instance is created for each particle in the swarm. A Pyomo
        solver factory is created with the Pyro solver manager.
        
        :param solver: (string) solver name. glpk, cplex, cbc, gurobi.
        :param links_dir: (directory) this directoy must contain three csv files:
                - `links.csv` the *network* with columns:
                    `['i','j','k','cost','amplitude','lower_bound','upper_bound']`
                - `inflows.csv` the *inflows* for entire period of analysis with columns:
                    `['date','j','hist',<insert as many columns with alternative flow sequences>]`
                - `sr_dict.json` the *reservoirs* for penalty search in the format of:
                    `{
                        "<reservoir id (e.g. SR_DNP)>":{
                            "eop_init":<initial storage (float)>,
                            "lb":<minimum storage (float)> ,
                            "ub":<maximum carryover capactiy (float)>,
                            "type": <1:COSVF or 2:Linear (integer)>,
                            "cosvf_param_index": <index(ices) to cosvf_pminmax PSO array (list)>
                            "k_count":<number of piecewise links to fit to quadratic cosvf (integer)>
                        }
                    }`
        :param flow_column: (string) column name of flows to use from the inflow.csv
        :param n_particles: (int) number of particles in the swarm
        :param options: (dict) a dictionary with keys `{'c1', 'c2', 'w'}` containing the parameters 
            for the PSO algorithm:
                * c1 : (float) cognitive parameter
                * c2 : (float) social parameter
                * w : (float) inertia parameter
        :param lower_bounds: (list) set the particles' lower bounds (required)
        :param upper_bounds: (list) optional list of the upper bounds on particle positions
        :param velocity_clamp: (tuple) optional tuple of size 2 where the first entry is the 
            minimum velocity and the second entry is the maximum velocity. It sets the limits 
            for velocity clamping.
        
        :returns: COSVF-PSO swarm and a CALVIN model for each particle
        """
        self.n_particles = n_particles
        self.options = options
        self.clamp = velocity_clamp
        self.dimensions = len(lower_bounds)
        self.links_dir = links_dir
        self.results_dir = os.path.join(links_dir, 'results', flow_column)

        # load default annual network
        self.links = pd.read_csv(os.path.join(links_dir, 'links.csv'))

        # load full time series of monthly inflows
        self.inflows = load_lf_inflows(links_dir, flow_column)
        self.wy_start = int(min(self.inflows.index.year)) + 1
        self.wy_end = int(max(self.inflows.index.year))

        # load sr dictionary
        self.sr_dict = load_sr_dict(links_dir)

        # Initialize Calvin model
        self.calvin_instances = []
        for i in range(n_particles):
            # Initialize Calvin Pyomo models (use lower bounds just as temporary stand in)
            calvin = CALVIN(
                linksfile=self.links,
                cosvf_pminmax=lower_bounds,
                sr_dict=self.sr_dict,
                inflows=self.inflows)
            calvin.create_pyomo_model()
            self.calvin_instances.append(calvin)

        # initialize pyomo solver factory
        self.cosvf_opt = SolverFactory(solver, threads=2)
        # instantiate a pyro solver manager
        try:
            self.cosvf_solver_manager = SolverManagerFactory('pyro')
            # set directory for temporary pyomo solver files
            self.temp_dir = os.path.join(tempfile.gettempdir(),'pyomo',flow_column)
            if not os.path.isdir(self.temp_dir):
                os.makedirs(self.temp_dir)
            TempfileManager.tempdir = self.temp_dir
        except:
            print("Failed to create solver manager. Restart the Pyro server.")
            sys.exit(1)

        # suppress pyomo warnings about infeasible models
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)

        # assign particle position bounds
        ub = np.zeros(self.dimensions) if upper_bounds is None else upper_bounds
        self.lb = np.repeat(np.array(lower_bounds)[np.newaxis, :], n_particles, axis=0)
        self.ub = np.repeat(np.array(ub)[np.newaxis, :], n_particles, axis=0)

        # initialize named tuple for populating the history list
        self.ToHistory = namedtuple(
            "ToHistory",
            [
                "position",
                "cost",
                "velocity",
            ],
        )

        # initialize particle history lists
        self.pos_hist = pd.DataFrame()
        self.cost_hist = pd.DataFrame()
        self.vel_hist = pd.DataFrame()

        # initialize the swarm
        position = self.generate_position()
        velocity = self.generate_velocity()
        self.swarm = Swarm(position, velocity, options=self.options)

        # assign name to cosvf_pso object
        self.name = __name__


    def populate_history(self, hist, iteration):
        """
        Populate all history dataframe

        :param hist: (collections.namedtuple)  Must be of the same type as self.ToHistory
        :param iteration: (int) iteration number

        :returns nothing but modifies self.cost_hist, .pos_hist, and .vel_hist
        """
        p_list = list(np.array(range(self.n_particles)) + 1)
        sr_list = list(self.sr_dict.keys())
        param = ['pmin', 'pmax'] #TODO change to accomodate linear (single param)

        p_df = pd.DataFrame(
            {'iteration': np.repeat(iteration, len(sr_list) * len(param) * len(p_list)),
             'particle': np.repeat(p_list, len(sr_list) * len(param)),
             'sr': list(np.repeat(sr_list, len(param))) * len(p_list),
             'param': param * len(sr_list) * len(p_list)})

        self.cost_hist = self.cost_hist.append(
            pd.DataFrame({'iteration': iteration, 'particle': p_list,
                          'cost': hist.cost}))

        pos = pd.melt(pd.DataFrame(hist.position).T)['value']
        self.pos_hist = self.pos_hist.append(pd.concat([p_df, pos], axis=1))

        vel = pd.melt(pd.DataFrame(hist.velocity).T)['value']
        self.vel_hist = self.vel_hist.append(pd.concat([p_df, vel], axis=1))


    def generate_position(self, initial=None):
        """
        Generate a new particle position
      
        :param initial: (boolean) If true, generates random position for all particles within initial 
            passed position and the upper and lower bounds. If false, generates a single random position 
            for single particle regeneration

        :returns pos: (numpy.ndarray) swarm matrix of shape (n_particles, n_dimensions)
        """
        pos = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.n_particles, self.dimensions))

        lb = self.lb.copy()
        for i in range(self.n_particles):
            for j in range(1, len(self.lb[0]), 2):
                lb[i][j - 1] = max(pos[i][j], self.lb[i][j - 1])
        ltb = np.nonzero(pos < lb)
        # and recalculate the pmin if necessary:
        pos[ltb] = lb[ltb] + np.random.uniform(1, (0 - self.lb[ltb])/2)

        return pos


    def generate_velocity(self):
        """
        Initialize a velocity vector

        :returns velocity: (numpy.ndarray) velocity matrix of shape (n_particles, dimensions)
        """
        min_velocity, max_velocity = (0, 1) if self.clamp is None else self.clamp

        velocity = (max_velocity - min_velocity) * np.random.random_sample(
            size=(self.n_particles, self.dimensions)) + min_velocity

        return velocity


    def optimize(self, iters):
        """
        Optimize the swarm

        :param iters: (int) number of iterations to conduct COSVF-PSO

        :returns: nothing, but saves COSVF-PSO result history (partcile position, cost, and velocity) 
            to three csv files in the `self.results_dir`
        """

        # Populate memory of the handlers
        self.memory = self.swarm.position
        self.swarm.pbest_cost = np.full(self.n_particles, np.inf)
        self.swarm.best_cost = np.inf
        best_cost_yet_found = np.NINF
        self.swarm.feasible = np.full(self.n_particles, False)

        for i in range(iters):
            # Optimize particles
            self.solve_cosvf_factory()

            # Determine global best and personal best
            self.compute_pbest()
            self.compute_gbest()

            # Save to history
            self.populate_history(self.ToHistory(
                position=self.swarm.position,
                cost=self.swarm.current_cost,
                velocity=self.swarm.velocity), i + 1)

            print('cost improvement: {}'.format(np.abs(self.swarm.best_cost - best_cost_yet_found)), flush=True)

            print('iteration: {}'.format(i), flush=True)
            print('pminmax: {}'.format(self.swarm.position.astype(int).tolist()), flush=True)
            print('pminmax_cost: {}'.format(self.swarm.current_cost.astype(int).tolist()), flush=True)
            print('pbest_cost: {}'.format(self.swarm.pbest_cost.astype(int).tolist()), flush=True)
            print('pbest_pminmax: {}'.format(self.swarm.pbest_pos.astype(int).tolist()), flush=True)

            # Perform velocity updates
            self.swarm.velocity = self.compute_velocity()

            print('flight: {}'.format(self.swarm.velocity.astype(int).tolist()), flush=True)
            print('best_cost: {}'.format(self.swarm.best_cost.astype(int)), flush=True)
            print('best_pminmax: {}'.format(self.swarm.best_pos.astype(int).tolist()), flush=True)

            # Perform position updates
            self.swarm.position = self.compute_position()

            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost

        # need to remove results directory since write **appends** data to files
        if os.path.isdir(self.results_dir): shutil.rmtree(self.results_dir)
        print('postprocessing to {}...'.format(self.results_dir))
        self.solve_cosvf_factory(final=True)


        psoHistFiles = ['pso_cost_history', 'pso_position_history', 'pso_velocity_history']
        for idx, hist in enumerate([self.cost_hist, self.pos_hist, self.vel_hist]):
            hist.to_csv(
                os.path.join(self.results_dir, '{}.csv'.format(psoHistFiles[idx])), index=False, header=True
            )

        # clean up temporary pyomo files
        shutil.rmtree(self.temp_dir)

        # job done.


    def compute_pbest(self):
        """
        Update the personal best score of the swarm by comparing the (1) cost of the current positions 
        and the (2) personal bests each particle attained in the swarm.

        :returns: nothing, but modifies COSVF-PSO swarm `pbest_cost`, `pbest_pos` and `feasible`
        """
        current_feasible = self.swarm.current_cost < 0

        for p in range(self.n_particles):
            # particle is feasible and it was feasible in the past:
            if (current_feasible[p] == True) and (self.swarm.feasible[p] == True):
                if self.swarm.current_cost[p] > self.swarm.pbest_cost[p]:
                    self.swarm.pbest_cost[p] = self.swarm.current_cost[p].copy()
                    self.swarm.pbest_pos[p] = self.swarm.position[p].copy()
            # particle is feasible and last position was infeasible:
            elif (current_feasible[p] == True) and (self.swarm.feasible[p] == False):
                self.swarm.pbest_cost[p] = self.swarm.current_cost[p].copy()
                self.swarm.pbest_pos[p] = self.swarm.position[p].copy()
            # particle is infeasible and was never feasible:
            elif (current_feasible[p] == False) and (self.swarm.feasible[p] == False):
                if self.swarm.current_cost[p] < self.swarm.pbest_cost[p]:
                    self.swarm.pbest_cost[p] = self.swarm.current_cost[p].copy()
                    self.swarm.pbest_pos[p] = self.swarm.position[p].copy()
            # (no update if particle is now infeasible but was previously feasible)

        self.swarm.feasible = current_feasible.copy()


    def compute_gbest(self):
        """
        Update the global best particle position using a global topolgy (best of all particles in swarm)

        :returns: nothing, but modifies COSVF_PSO swarm's `best_cost` and `best_pos`
        """
        current_feasible = np.where(self.swarm.feasible == True)[0]
        if (current_feasible.size == 0) and (np.all(self.swarm.pbest_cost > 0)):
            if self.swarm.best_cost > np.min(self.swarm.pbest_cost):
                self.swarm.best_cost = np.min(self.swarm.pbest_cost)
                best_pos_idx = np.where(self.swarm.pbest_cost == self.swarm.best_cost)[0]
                self.swarm.best_pos = self.swarm.pbest_pos[best_pos_idx].copy()

        elif current_feasible.size > 0:
            if ((self.swarm.best_cost > 0) or
                    (self.swarm.best_cost < np.max(self.swarm.pbest_cost[current_feasible]))):
                self.swarm.best_cost = np.max(self.swarm.pbest_cost[current_feasible]).copy()
                best_pos_idx = np.where(self.swarm.pbest_cost == self.swarm.best_cost)[0]
                self.swarm.best_pos = self.swarm.pbest_pos[best_pos_idx].copy()


    def compute_gbest_ring(self, p, k):
        """
        Update the global best using a ring-like neighborhood approach. This uses the cKDTree method 
        from `scipy` to obtain the nearest eighbors.

        ** NOT YET FULLY IMPLEMENTED/TESTED ***

        :param p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        :param k: int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`

        :returns: nothing, but modifies COSVF_PSO swarm's `best_cost` and `best_pos`
        """
        # Check if the topology is static or not and assign neighbors
        if (self.static and self.neighbor_idx is None) or not self.static:
            # Obtain the nearest-neighbors for each particle
            tree = cKDTree(self.swarm.position)
            _, self.neighbor_idx = tree.query(self.swarm.position, p=p, k=k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if k == 1:
            # The minimum index is itself, no mapping needed.
            self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
            best_neighbor = np.arange(self.swarm.n_particles)
        else:
            idx_min = self.swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
            best_neighbor = self.neighbor_idx[
                np.arange(len(self.neighbor_idx)), idx_min
            ]
        # Obtain best cost and position
        self.swarm.best_cost = np.min(self.swarm.pbest_cost[best_neighbor])
        self.swarm.best_pos = self.swarm.pbest_pos[best_neighbor]


    def compute_velocity(self):
        """
        Update the velocity matrix.

        :returns updated_velocity: (numpy.ndarray) Updated velocity matrix
        """

        # Prepare parameters
        swarm_size = self.swarm.position.shape
        c1 = self.swarm.options["c1"]
        c2 = self.swarm.options["c2"]
        w = self.swarm.options["w"]

        pbest_sign = np.where(self.swarm.pbest_pos > self.swarm.position, 1, -1)
        best_sign = np.where(self.swarm.best_pos > self.swarm.position, 1, -1)

        # Compute for cognitive and social terms
        cognitive = (
                c1
                * np.random.uniform(0, 1, swarm_size)
                * np.abs(self.swarm.pbest_pos - self.swarm.position) * pbest_sign
        )
        social = (
                c2
                * np.random.uniform(0, 1, swarm_size)
                * np.abs(self.swarm.best_pos - self.swarm.position) * best_sign
        )

        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * self.swarm.velocity) + cognitive + social
        if self.clamp:
            clamped_vel = temp_velocity
            min_velocity, max_velocity = self.clamp
            lower_than_clamp = clamped_vel <= min_velocity
            greater_than_clamp = clamped_vel >= max_velocity
            clamped_vel = np.where(lower_than_clamp, min_velocity, clamped_vel)
            clamped_vel = np.where(greater_than_clamp, max_velocity, clamped_vel)
            updated_velocity = clamped_vel
        else:
            updated_velocity = temp_velocity

        return updated_velocity

    def compute_position(self):
        """
        Set the particle to a new position based on velocity.

        If new position is outside permissible bounds, this method resets particles that 
        exceed the bounds to an intermediate position between the boundary and their earlier position. 
        Namely, it changes the coordinate of the out-of-bounds axis to: bound plus (lower) or 
        minus (upper) a uniform random number [0.5 to 4] * 0.05 * distance between the lower and upper bound

        :returns new_pos: (numpy.ndarray) New position-matrix
        """
        new_pos = self.memory.copy()
        new_pos += self.swarm.velocity

        # check any particles that violate upper bound on position
        gtb = np.nonzero(new_pos > self.ub)
        # and recalculate the pmin and pmax if necessary
        new_pos[gtb] = self.ub[gtb] - (np.abs(self.ub[gtb] - self.lb[gtb]) / 20) * np.random.uniform(0.5, 4)

        # pmax of new position cannot be less than its lower bound
        # also pmin of new position cannot be less negative than pmax
        lb = self.lb.copy()
        for i in range(self.n_particles):
            for j in range(1, len(self.lb[0]), 2):
                lb[i][j - 1] = max(new_pos[i][j], self.lb[i][j - 1])
        ltb = np.nonzero(new_pos < lb)
        # and recalculate the pmin if necessary:
        new_pos[ltb] = lb[ltb] + (np.abs(self.ub[ltb] - lb[ltb]) / 20) * np.random.uniform(0.5, 4)

        self.memory = new_pos.copy()

        return new_pos

    def solve_cosvf_factory(self, final=False):
        """
        Solve CALVIN instances in the COSVF-PSO swarm for period of analysis.

        Procedure: 
            1. Split/assign the swarm particle positions to the CALVIN instances
            2. Initialize the COSVF for the CALVIN instance (set the piecewise link penalties)
            3. Loop through the period of analysis, solving the CALVIN instance for each year and 
            accumulating the objective value or the constraint violation volume if infeasible.
            The end of period storage for each annual run is extracted and set as the initial value
            for the next year in the sequence with `cosvf_update_sequence`
            4. Collect final objective values (and constraint violations) as cost array for the 
            particles in the swarm. This is used to determine personal and best swarm positions
            and update the COSVF parameters for the next iteration.

        :param final: (boolean) Whether or not this is the last PSO iteration (determines write)
        
        :returns: nothing, but modifies COSVF-PSO swarm's `current_cost` numpy.ndarray
        """
        # initialize first year of each model with new cosvf parameters
        particle_split = np.squeeze(np.array_split(self.swarm.position, self.n_particles))
        if not final:
            for idx, particle in enumerate(particle_split):
                self.calvin_instances[idx].cosvf_pminmax = particle
                self.calvin_instances[idx].cosvf_init_sequence()
        else:
            # this is the final COSVF best run position
            self.calvin_instances[0].cosvf_pminmax = self.swarm.best_pos.tolist()[0]
            self.calvin_instances[0].cosvf_init_sequence()
            # use the first CALVIN instance to solve (but keep as list to satistfy solver factory logic)
            self.calvin_instances = self.calvin_instances[0:1] 

        # model dictionary for end-of-period storage
        eops = {}
        # model dictionary for accumulating annual objective function value
        objective_values = {}
        # fill with initial defaults
        for i in range(len(self.calvin_instances)):
            objective_values[i] = 0
            eops[i] = None

        for wy in range(self.wy_start, self.wy_end + 1):
            # maps models to a handle id
            action_handle_map = {}
            for idx,c in enumerate(self.calvin_instances):
                # continue to queue model solve if it's feasible
                if objective_values[idx] <= 0:
                    c.cosvf_update_sequence(eops[idx], wy)
                    action_handle = self.cosvf_solver_manager.queue(
                        c.model, opt=self.cosvf_opt, warmstart=False, tee=False, keepfiles=False)
                    action_handle_map[action_handle] = idx

            # retrieve solutions
            for each in range(len(action_handle_map)):

                try:
                    this_action_handle = self.cosvf_solver_manager.wait_any()
                    solved_name = action_handle_map[this_action_handle]
                    result = self.cosvf_solver_manager.get_results(this_action_handle)
                except:
                    print('solver error')
                    continue
                # grab CALVIN instance according to its solver handle
                c = self.calvin_instances[solved_name]
                # determine objective value and eop storage (or constriant violation if infeasible)
                if ((result.solver.status == SolverStatus.ok) and
                        (result.solver.termination_condition == TerminationCondition.optimal)):
                    # load model results
                    m = c.model
                    # accumulate total cost
                    objective_values[solved_name] += value(m.total())
                    # eop dictionary for initial condition of next year in sequence
                    eop = {}
                    for sr in self.sr_dict:
                        eop[sr] = 0
                        # accumulate total storage over k piecewise links
                        for k in range(self.sr_dict[sr]['k_count']):
                            eop[sr] += m.X[('{}.{}-09-30'.format(sr,self.wy_start), 'FINAL', k)].value
                    # add eop dictionary to model dictionary for eops
                    eops[solved_name] = eop
                    # postprocessing and saving (if desired)
                    if final: postprocess(c.df, m, resultdir=self.results_dir, annual=True, year=wy)

                elif result.solver.status == SolverStatus.warning:
                    # if infeasible, kick out of loop and return constraint violation magnitudes
                    eq, ineq = c.calc_infeasible_constraints()
                    objective_values[solved_name] = np.add(eq, ineq)

                else:
                    # Something else is wrong
                    objective_values[solved_name] = np.inf

        self.swarm.current_cost = np.array(list(objective_values.values()))

@attrs
class Swarm(object):
    """
    The COSVF-PSO swarm object

    Attributes
    ----------
    :attribute position: (numpy.ndarray) position-matrix at a given timestep of shape `(n_particles, dimensions)`
    :attribute velocity: (numpy.ndarray) velocity-matrix at a given timestep of shape `(n_particles, dimensions)`
    :attribute n_particles : (int) number of particles in a swarm.
    :attribute dimensions: (int) number of dimensions in a swarm.
    :attribute options: (dict) options that govern a swarm's behavior.
    :attribute pbest_pos: (numpy.ndarray) personal best positions of each particle of shape `(n_particles, dimensions)`
    :attribute best_pos: (numpy.ndarray) best position found by the swarm of shape `(dimensions, )` for
          the global topology and `(dimensions, particles)` for the other topologies
    :attribute pbest_cost: (numpy.ndarray) personal best costs of each particle of shape `(n_particles, )`
    :attribute best_cost: (float) best cost found by the swarm
    :attribute current_cost: (numpy.ndarray) current cost found by the swarm of shape `(n_particles, dimensions)`
      """
    # Required attributes
    position = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    velocity = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    # With defaults
    n_particles = attrib(type=int, validator=instance_of(int))
    dimensions = attrib(type=int, validator=instance_of(int))
    options = attrib(type=dict, default={}, validator=instance_of(dict))
    pbest_pos = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    best_pos = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))
    pbest_cost = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))
    best_cost = attrib(type=float, default=np.inf, validator=instance_of((int, float)))
    current_cost = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))

    @n_particles.default
    def n_particles_default(self):
        return self.position.shape[0]

    @dimensions.default
    def dimensions_default(self):
        return self.position.shape[1]

    @pbest_pos.default
    def pbest_pos_default(self):
        return self.position
