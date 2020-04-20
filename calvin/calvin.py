import sys, os, shutil, re
from pyomo.environ import *
from pyomo.opt import SolverStatus, SolverFactory, TerminationCondition
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.core import Constraint, value
import numpy as np
import pandas as pd
from math import fabs
from .postprocessor import *
from .utils import *


class CALVIN:


    def __init__(self, linksfile, ic=None, cosvf_pminmax=None, sr_dict=None, inflows=None):
        """
        Initialize CALVIN model object.

        :param linksfile: (string or pandas dataframe) CSV file containing network link information
        :param ic: (dict) *fixed multipliers LF-mode*:, the 
            initial storage conditions for surface reservoirs 
        :param cosvf_pminmax (array): *COSVF LF-mode*: a  list of parameters to define 
            quadratic carryover penalty curves for surface water reservoirs or linear storage 
            penalties on groundwater reservoirs 
        :param sr_dict: (dict) *COSVF LF-mode*: the *reservoirs* for which penalty curves are used
            {
                "<reservoir id (e.g. SR_DNP)>":{
                    "eop_init":<initial storage (float)>,
                    "lb":<minimum storage (float)> ,
                    "ub":<maximum carryover capactiy (float)>,
                    "type": <1:COSVF or 2:Linear (integer)>,
                    "cosvf_param_index": <index(ices) to cosvf_pminmax PSO array (list)>
                    "k_count":<number of piecewise links to fit to quadratic cosvf (integer)>
                }
            }
        :param inflows: (pandas dataframe) *COSVF LF-mode*: dataframe of inflows

        :returns: CALVIN model object
        """
        # load links
        self.df = pd.read_csv(linksfile) if type(linksfile) is str else linksfile.copy()

        # COSVF links if Pmin and Pmax is passed
        if cosvf_pminmax is not None:
            self.cosvf_pminmax = cosvf_pminmax
            self.sr_dict = sr_dict
            self.inflows = inflows
            self.wy_start = int(min(self.inflows.index.year)) + 1
            self.wy_end = int(max(self.inflows.index.year))
            self.cosvf_links_create()

        # unique link index
        self.df['link'] = self.df.i.map(str) + '_' + self.df.j.map(str) + '_' + self.df.k.map(str)
        self.df.set_index('link', inplace=True)

        # filename w/o extension
        self.linksfile = os.path.splitext(linksfile)[0] if type(linksfile) is str else 'ann'

        # SR stats for eop constant multiplier
        sr_stats = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'data/california-network/SR_stats.csv'),
            index_col=0).to_dict()
        self.min_storage = sr_stats['min']
        self.max_storage = sr_stats['max']

        # apply an initial storage value if running annual mode with fixed eop multiplier
        if ic: self.apply_ic(ic)

        # add ag sinks to get rid of surplus water
        self.add_ag_region_sinks()

        # ensure that month-to-month storage links are correct
        self.fix_hydropower_lbs()

        # create a unique node dataframe and a links list
        self.nodes = pd.unique(self.df[['i','j']].values.ravel()).tolist()
        self.links = list(zip(self.df.i,self.df.j,self.df.k))

        # make sure things aren't broken
        self.networkcheck() 


    def apply_ic(self, ic):
        """
        Set initial storage conditions.

        :param ic: (dict) initial storage values

        :returns: nothing, but modifies the model object
        """
        for k in ic:
            ix = (self.df.i.str.contains('INITIAL') &
                    self.df.j.str.contains(k))
            self.df.loc[ix, ['lower_bound','upper_bound']] = ic[k]


    def inflow_multiplier(self, x):
        """
        Multiply all network inflows by a constant.

        :param x: (float) value to multiply inflows

        :returns: nothing, but modifies the model object
        """

        ix = self.df.i.str.contains('INFLOW')
        self.df.loc[ix, ['lower_bound','upper_bound']] *= x


    def eop_constraint_multiplier(self, x):
        """
        Set end-of-period storage constraints as a fraction of maximum
        available storage. Used in limited foresight optimization
        with fixed multipliers.

        :param x: (float) fraction of maximum storage to set lower bound

        :returns: nothing, but modifies the model object
        """

        for k in self.max_storage:
            ix = (self.df.i.str.contains(k) &
                    self.df.j.str.contains('FINAL'))
            lb = self.min_storage[k] + (self.max_storage[k]-self.min_storage[k])*x
            self.df.loc[ix,'lower_bound'] = lb
            self.df.loc[ix,'upper_bound'] = self.max_storage[k]
  

    def no_gw_overdraft(self):
        """
        Impose constraints to prevent groundwater overdraft

        (not currently implemented)
        """
        pass


    def networkcheck(self):
        """
        Confirm constraint feasibility for the model object.

        (No inputs or outputs)

        :raises: ValueError when infeasibilities are identified.
        """

        nodes = self.nodes
        links = self.df.values

        num_in = {n: 0 for n in nodes}
        num_out = {n: 0 for n in nodes}
        lb_in = {n: 0 for n in nodes}
        lb_out = {n: 0 for n in nodes}
        ub_in = {n: 0 for n in nodes}
        ub_out = {n: 0 for n in nodes}

        # loop over links
        for l in links:
            lb = float(l[5])
            ub = float(l[6])
            num_in[l[1]] += 1
            lb_in[l[1]] += lb
            ub_in[l[1]] += ub
            num_out[l[0]] += 1
            lb_out[l[0]] += lb
            ub_out[l[0]] += ub

            if lb > ub:
                raise ValueError('lb > ub for link %s' % (l[0]+'-'+l[1]))

        for n in nodes:
            if num_in[n] == 0 and n not in ['SOURCE','SINK']:
                raise ValueError('no incoming link for ' + n)
            if num_out[n] == 0 and n not in ['SOURCE','SINK']:
                raise ValueError('no outgoing link for ' + n)

            if ub_in[n] < lb_out[n]:
                raise ValueError('ub_in < lb_out for %s (%d < %d)' % (n, ub_in[n], lb_out[n]))
            if lb_in[n] > ub_out[n]:
                raise ValueError('lb_in > ub_out for %s (%d > %d)' % (n, lb_in[n], ub_out[n]))


    def add_ag_region_sinks(self):
        """
        Get rid of surplus water at no cost from agricultural regions. Called internally 
        when model is initialized.

        :returns: nothing, but modifies the model object
        """

        df = self.df
        links = df[df.i.str.contains('HSU') & ~df.j.str.contains('DBUG')].copy(deep=True)
        if not links.empty:
            maxub = links.upper_bound.max()
            links.j = links.apply(lambda l: 'SINK.'+l.i.split('.')[1], axis=1)
            links.cost = 0.0
            links.amplitude = 1.0
            links.lower_bound = 0.0
            links.upper_bound = maxub
            links['link'] = links.i.map(str) + '_' + links.j.map(str) + '_' + links.k.map(str)
            links.set_index('link', inplace=True)
            self.df = self.df.append(links.drop_duplicates())


    def fix_hydropower_lbs(self):
        """
        Fix lower bound constraints on piecewise hydropower links. Storage piecewise 
        links > 0 should have 0.0 lower bound, and the k=0 pieces should always have 
        lb = dead pool.

        :returns: nothing, but modifies the model object
        """

        def get_lb(link):
            if link.i.split('.')[0] == link.j.split('.')[0]:
                if link.k > 0:
                    return 0.0
                elif link.i.split('.')[0] in self.min_storage:
                    return min(self.min_storage[link.i.split('.')[0]], link.lower_bound)
            return link.lower_bound

        ix = (self.df.i.str.contains('SR_') & self.df.j.str.contains('SR_'))
        self.df.loc[ix, 'lower_bound'] = self.df.loc[ix].apply(get_lb, axis=1)


    def remove_debug_links(self):
        """
        Remove debug links from model object.

        :returns: dataframe of links, excluding debug links.
        """
        df = self.df
        ix = df.index[df.index.str.contains('DBUG')]
        df.drop(ix, inplace=True, axis=0)
        self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
        self.links = list(zip(df.i,df.j,df.k))
        return df


    def create_pyomo_model(self, debug_mode=False, debug_cost=2e7):
        """
        Use link data to create Pyomo model (constraints and objective function)
        But do not solve yet.

        :param debug_mode: (boolean)  Whether to run in debug mode. 
            Use when there may be infeasibilities in the network.
        :param debug_cost: (float) When in debug mode, assign this cost ($/AF) to flow on debug links. 
            This should be an arbitrarily high number.

        :returns: nothing, but creates the model object (self.model)
        """

        # work on a local copy of the dataframe
        if not debug_mode and self.df.index.str.contains('DBUG').any():
            # previously ran in debug mode, but now done
            df = self.remove_debug_links()
            df.to_csv(self.linksfile + '-final.csv')
        else:
            df = self.df

        print('Creating Pyomo Model (debug=%s)' % debug_mode, flush=True)

        model = ConcreteModel()

        model.N = Set(initialize=self.nodes)
        model.k = Set(initialize=range(15))
        model.A = Set(within=model.N*model.N*model.k,
                        initialize=self.links, ordered=True)
        model.source = Param(initialize='SOURCE')
        model.sink = Param(initialize='SINK')

        def init_params(p):
            if p == 'cost' and debug_mode:
                return (lambda model,i,j,k: debug_cost
                    if ('DBUG' in str(i)+'_'+str(j))
                    else 1.0)
            else:
                return lambda model,i,j,k: df.loc[str(i)+'_'+str(j)+'_'+str(k)][p]

        model.u = Param(model.A, initialize=init_params('upper_bound'), mutable=True)
        model.l = Param(model.A, initialize=init_params('lower_bound'), mutable=True)
        model.a = Param(model.A, initialize=init_params('amplitude'))
        model.c = Param(model.A, initialize=init_params('cost'), mutable=True)

        # The flow over each arc
        model.X = Var(model.A, within=Reals)

        # Minimize total cost
        def obj_fxn(model):
            return sum(model.c[i,j,k]*model.X[i,j,k] for (i,j,k) in model.A)
        model.total = Objective(rule=obj_fxn, sense=minimize)

        # Enforce an upper bound limit on the flow across each arc
        def limit_rule_upper(model, i, j, k):
            return model.X[i,j,k] <= model.u[i,j,k]
        model.limit_upper = Constraint(model.A, rule=limit_rule_upper)

        # Enforce a lower bound limit on the flow across each arc
        def limit_rule_lower(model, i, j, k):
            return model.X[i,j,k] >= model.l[i,j,k]
        model.limit_lower = Constraint(model.A, rule=limit_rule_lower)

        # To speed up creating the mass balance constraints, first
        # create dictionaries of arcs_in and arcs_out of every node
        # These are NOT Pyomo data, and Pyomo does not use "model._" at all
        arcs_in = {}
        arcs_out = {}

        def arc_list_hack(model, i,j,k):
            if j not in arcs_in:
                arcs_in[j] = []
            arcs_in[j].append((i,j,k))

            if i not in arcs_out:
                arcs_out[i] = []
            arcs_out[i].append((i,j,k))
            return [0]

        model._ = Set(model.A, initialize=arc_list_hack)

        # Enforce flow through each node (mass balance)
        def flow_rule(model, node):
            if node in [value(model.source), value(model.sink)]:
                return Constraint.Skip
            outflow  = sum(model.X[i,j,k]/model.a[i,j,k] for i,j,k in arcs_out[node])
            inflow = sum(model.X[i,j,k] for i,j,k in arcs_in[node])
            return inflow == outflow
        model.flow = Constraint(model.N, rule=flow_rule)

        model.dual = Suffix(direction=Suffix.IMPORT)

        self.model = model


    def solve_pyomo_model(self, solver='glpk', nproc=1, debug_mode=False, maxiter=10):
        """
        Solve Pyomo model (must be called after create_pyomo_model)

        :param solver: (string) solver name. glpk, cplex, cbc, gurobi.
        :param nproc: (int) number of processors. 1=serial.
        :param debug_mode: (boolean) Whether to run in debug mode. 
            Use when there may be infeasibilities in the network.
        :param maxiter: (int) maximum iterations for debug mode.

        :returns: nothing, but assigns results to self.model.solutions.

        :raises: RuntimeError, if problem is found to be infeasible.
        """
        opt = SolverFactory(solver)

        if nproc > 1 and solver is not 'glpk':
            opt.options['threads'] = nproc

        if debug_mode:
            run_again = True
            i = 0
            vol_total = 0

            while run_again and i < maxiter:
                print('-----Solving Pyomo Model (debug=%s)' % debug_mode)
                self.results = opt.solve(self.model)
                print('Finished. Fixing debug flows...')
                run_again,vol = self.fix_debug_flows()
                i += 1
                vol_total += vol

            if run_again:
                print(('Warning: Debug mode maximum iterations reached.'
                    ' Will still try to solve without debug mode.'))
            else:
                print('All debug flows eliminated (iter=%d, vol=%0.2f)' % (i,vol_total))

        else:
            print('-----Solving Pyomo Model (debug=%s)' % debug_mode)
            self.results = opt.solve(self.model, tee=False)

            if self.results.solver.termination_condition == TerminationCondition.optimal:
                print('Optimal Solution Found (debug=%s).' % debug_mode)
                self.model.solutions.load_from(self.results)
            
            else:
                raise RuntimeError('Problem Infeasible. Run again starting from debug mode.')


    def fix_debug_flows(self, tol=1e-7):
        """
        Find infeasible constraints where debug flows occur. Fix them by either 
        raising the UB, or lowering the LB.

        :param tol: (float) Tolerance to identify nonzero debug flows
        :returns run_again: (boolean) whether debug mode needs to run again

        :returns vol: (float) total volume of constraint changes also modifies the model object.
        """

        df, model = self.df, self.model
        dbix = (df.i.str.contains('DBUGSRC') | df.j.str.contains('DBUGSNK'))
        debuglinks = df[dbix].values

        run_again = False
        vol_total = 0

        for dbl in debuglinks:
            s = tuple(dbl[0:3])

            if model.X[s].value > tol:
                run_again = True

                # if we need to get rid of extra water,
                # raise some upper bounds (just do them all)
                if 'DBUGSNK' in dbl[1]:
                    raiselinks = df[(df.i == dbl[0]) & ~ df.j.str.contains('DBUGSNK')].values

                    for l in raiselinks:
                        s2 = tuple(l[0:3])
                        iv = model.u[s2].value
                        v = model.X[s].value*1.2
                        model.u[s2].value += v
                        vol_total += v
                        print('%s UB raised by %0.2f (%0.2f%%)' % (l[0]+'_'+l[1], v, v*100/iv))
                        df.loc['_'.join(str(x) for x in l[0:3]), 'upper_bound'] = model.u[s2].value

                # if we need to bring in extra water
                # this is a much more common problem
                # want to avoid reducing carryover requirements. look downstream instead.
                max_depth = 10

                if 'DBUGSRC' in dbl[0]:
                    vol_to_reduce = max(model.X[s].value*1.2, 0.5)
                    print('Volume to reduce: %.2e' % vol_to_reduce)

                    children = [dbl[1]]
                    for i in range(max_depth):
                        children += df[df.i.isin(children) & ~ df.j.str.contains('DBUGSNK')].j.tolist()
                    children = set(children)
                    reducelinks = (df[df.i.isin(children) & (df.lower_bound > 0)]
                        .sort_values(by='lower_bound', ascending=False).values)

                    if reducelinks.size == 0:
                        raise RuntimeError(('Not possible to reduce LB on links with origin %s by volume %0.2f' %
                                        (dbl[1],vol_to_reduce)))

                    for l in reducelinks:
                        s2 = tuple(l[0:3])
                        iv = model.l[s2].value
                        dl = model.dual[model.limit_lower[s2]] if s2 in model.limit_lower else 0.0

                        if iv > 0 and vol_to_reduce > 0 and dl > 1e6:
                            v = min(vol_to_reduce, iv)
                            # don't allow big reductions on carryover links
                            carryover = ['SR_', 'INITIAL', 'FINAL', 'GW_']
                            if any(c in l[0] for c in carryover) and any(c in l[1] for c in carryover):
                                v = min(v, max(25.0, 0.1*iv))
                            model.l[s2].value -= v
                            vol_to_reduce -= v
                            vol_total += v
                            print('%s LB reduced by %.2e (%0.2f%%). Dual=%.2e' % (l[0]+'_'+l[1], v, v*100/iv, dl))
                            df.loc['_'.join(str(x) for x in l[0:3]), 'lower_bound'] = model.l[s2].value

                        if vol_to_reduce == 0:
                            break

                    if vol_to_reduce > 0:
                        print('Debug -> %s: could not reduce full amount (%.2e left)' % (dbl[1],vol_to_reduce))

        self.df, self.model = df, model

        return run_again,vol_total


    def calc_infeasible_constraints(self, tol=1.):
        """
        Iterates over an infeasible CALVIN model and accumulates total volume of 
        equality and inequality constraint violations.

        :param m: (object) CALVIN model obejct to iterate constraints over.
        :param tol: (float) Volume that constraint violation must be above to be counted.
            Some constraint violations in the solver are within the model solver tolerance and 
            therefore very, very small (10e-20) and do not trigger a model infeasibility; those 
            would still be 'counted' here if not screened out by the tolerance threshold.

        :returns eq, ineq: (float) Total volume of equality amd inequality constraint violations
        """

        ineq = 0
        eq = 0

        # Iterate through all active constraints on the model
        for constr in self.model.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):

            constr_body_value = value(constr.body, exception=False)
            constr_lb_value = value(constr.lower, exception=False)
            constr_ub_value = value(constr.upper, exception=False)

            constr_undefined = False
            equality_violated = False
            lb_violated = False
            ub_violated = False

            if constr_body_value is None:
                # Undefined constraint body value due to missing variable value
                constr_undefined = True
                pass
            else:
                # Check for infeasibility
                if constr.equality:
                    if fabs(constr_lb_value - constr_body_value) >= tol:
                        equality_violated = True
                else:
                    if constr.has_lb() and constr_lb_value - constr_body_value >= tol:
                        lb_violated = True
                    if constr.has_ub() and constr_body_value - constr_ub_value >= tol:
                        ub_violated = True

            if not any((constr_undefined, equality_violated, lb_violated, ub_violated)):
                # constraint is fine. skip to next constraint
                continue

            if constr.equality:
                eq += fabs(constr_lb_value - constr_body_value)

            elif constr.has_lb():
                ineq += fabs(constr_lb_value - constr_body_value)

            elif constr.has_ub():
                ineq += fabs(constr_body_value - constr_ub_value)

        return eq, ineq


    def cosvf_link_costs(self, sr):
        """
        Create piecewise costs for penalties on end of period storage.
        
        :param sr: (str) reservoir id

        :returns sr_b: (list) storage link breakpoints
                 sr_k: (list) storage link costs
        """
        # check sr type (1:quadratic COSVF; 2:linear penalty)
        if self.sr_dict[sr]['type']==1:

            # construct cosvf from params
            cosvfx, cosvfy = cosvf_fit(
                pmin=self.cosvf_pminmax[self.sr_dict[sr]['cosvf_param_index'][0]], 
                pmax=self.cosvf_pminmax[self.sr_dict[sr]['cosvf_param_index'][1]],
                eop_min=self.sr_dict[sr]['lb'],
                eop_max=self.sr_dict[sr]['ub'])

            # get piecewise storage breakpoints and penalty slopes
            sr_b, sr_k = cosvf_pwlf(cosvfx, cosvfy, self.sr_dict[sr]['k_count'])

        # :TODO: implement linear penalty

        return sr_b, sr_k

    def cosvf_links_create(self):
        """
        Create links for the storage nodes that define the carryover penalties. 

        :returns: nothing, but modifies links dataframe
        """

        # loop through reservoirs passing new pmin and pmax parameters
        for sr in self.sr_dict:

            # edit k, ub, and lb in calvin sr final nodes
            sr_links = self.df[(self.df.i.str.contains(sr)) & (self.df.j.str.contains('FINAL'))].copy()
            self.df.drop(sr_links.index, inplace=True) # remove sr node so it's not duplicated

            # paste in the minimum capacity for k=0
            sr_links.loc[sr_links.k==0,'lower_bound'] = self.sr_dict[sr]['lb']
            sr_links = sr_links.append([sr_links]*(self.sr_dict[sr]['k_count']-1), ignore_index=True)

            # label each piecewise k (integers)
            sr_links.loc[:,'k'] = list(range(self.sr_dict[sr]['k_count']))

            # get penalty link costs and breakpoints
            sr_links.upper_bound, sr_links.cost = self.cosvf_link_costs(sr)

            # set minimum capacities to zero for all other k
            sr_links.loc[sr_links.k>0,'lower_bound'] = 0

            self.df = self.df.append(sr_links)


    def cosvf_solve(self, results_dir, solver='glpk', nproc=1):
        """
        Solve CALVIN model for period of analysis in annual COSVF mode
        
        """

        # create model
        self.create_pyomo_model()

        # declare solver
        opt = SolverFactory(solver)
        if nproc > 1 and solver is not 'glpk':
            opt.options['threads'] = nproc

        # initiate first year
        self.cosvf_init_sequence()

        # initialize objective function value accumulation
        objective_value = 0

        # set eop to null (to use sr_dict for the first year)
        eop = None

        # loop through years in sequence
        for wy in range(self.wy_start, self.wy_end + 1):

            self.cosvf_update_sequence(eop=eop, wy=wy)

            print('-----Solving Pyomo Model (wy=%d)' % wy)
            self.results = opt.solve(self.model, tee=False, keepfiles=False)

            if ((self.results.solver.status == SolverStatus.ok) and
                    (self.results.solver.termination_condition == TerminationCondition.optimal)):
                
                # load solution to model
                self.model.solutions.load_from(self.results)
                objective_value += value(self.model.total())
                print('Optimal Solution Found (obj. value=%d).' % objective_value)

                # eop dictionary for initial condition of next year in sequence
                eop = {}
                for sr in self.sr_dict:
                    eop[sr] = 0
                    # accumulate total storage over k piecewise links
                    for k in range(self.sr_dict[sr]['k_count']):
                        eop[sr] += self.model.X[('{}.{}-09-30'.format(sr,self.wy_start), 'FINAL', k)].value

                # postprocessing and saving
                postprocess(self.df, self.model, resultdir=results_dir, annual=True, year=wy)

            elif self.results.solver.status == SolverStatus.warning:
                # if infeasible, kick out of loop and return constraint violation magnitudes
                eq, ineq = self.calc_infeasible_constraints(self.model)
                print('Model is infeasible (wy=%f).' % wy)
                print('Inequality violation (TAF): %f' % ineq)
                print('Equality violation (TAF): %f' % eq)
                break

            else:
                # Something else is wrong
                print('Solver issue.')
                break


    def cosvf_init_sequence(self):
        """
        Initialize first year of the model in COSVF annual mode

        """
        # calculate new piecewise cosvf based on pmin and pmax and apply to sr links
        for sr in self.sr_dict:

            # get penalty link costs and breakpoints
            links_b, links_k  = self.cosvf_link_costs(sr)

            # assign piecewise cosvf to calvin model reservoir links
            for k in range(self.sr_dict[sr]['k_count'] - 1):
                self.model.c[('{}.{}-09-30'.format(sr,self.wy_start), 'FINAL', k)] = round(links_k[k], 4)
                self.model.u[('{}.{}-09-30'.format(sr,self.wy_start), 'FINAL', k)] = round(links_b[k], 4)

            # assign initial storage
            self.model.l[('INITIAL', '{}.{}-10-31'.format(sr,self.wy_start-1), 0)] = self.sr_dict[sr]['eop_init']
            self.model.u[('INITIAL', '{}.{}-10-31'.format(sr,self.wy_start-1), 0)] = self.sr_dict[sr]['eop_init']


    def cosvf_update_sequence(self, eop, wy):
        """
        Update initial storages and monthly inflows for the model in COSVF annual mode

        """
        # update initial storage condition
        if eop is not None:
            for sr in eop:
                self.model.l[('INITIAL', '{}.{}-10-31'.format(sr,self.wy_start-1), 0)] = round(eop[sr], 4)
                self.model.u[('INITIAL', '{}.{}-10-31'.format(sr,self.wy_start-1), 0)] = round(eop[sr], 4)

        # replace inflows for wy in sequence
        offset = wy - self.wy_start
        dates = pd.date_range('{}-10-31'.format(wy - 1), '{}-09-30'.format(wy), freq='M')
        for sr in self.sr_dict:
            for date in dates:
                inflow = self.inflows.loc[date, sr]
                fdate = str((date - pd.DateOffset(years=offset)).date())
                self.model.l[('INFLOW.{}'.format(fdate), '{}.{}'.format(sr, fdate), 0)] = round(inflow, 4)
                self.model.u[('INFLOW.{}'.format(fdate), '{}.{}'.format(sr, fdate), 0)] = round(inflow, 4)
