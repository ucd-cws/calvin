import os
import sys
import logging

from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy as np
import pandas as pd

class CALVIN():

  def __init__(self, linksfile, ic=None, log_name="calvin"):
    """
    Initialize CALVIN model object.

    :param linksfile: (string) CSV file containing network link information
    :param ic: (dict) Initial storage conditions for surface reservoirs
                only used for annual optimization
    :param log_name: A name for a logger - will be used to keep logs from different model runs separate in files.
                Defaults to "calvin", which results in a log file in the current working directory named "calvin.log".
                You can change this each time you instantiate the CALVIN class if you want to output separate logs
                for different runs. Otherwise, all results will be appended to the log file (not overwritten). If you
                run multiple copies of CALVIN simultaneously, make sure to change this, or you could get errors writing
                to the log file.

                Do not provide a full path to a log file here because this value is also used in a way that is *not* a
                file path. If being able to specify a full path is important for your workflow, please raise a GitHub
                issue. It could be supported, but there is no need at this moment.
    :returns: CALVIN model object
    """

    # set up logging code
    self.log = logging.getLogger(log_name)
    if not self.log.hasHandlers():  # hasHandlers will only be True if someone already called CALVIN with the same log_name in the same session
      self.log.setLevel("DEBUG")
      screen_handler = logging.StreamHandler(sys.stdout)
      screen_handler.setLevel(logging.INFO)
      screen_formatter = logging.Formatter('%(levelname)s - %(message)s')
      screen_handler.setFormatter(screen_formatter)
      self.log.addHandler(screen_handler)

      file_handler = logging.FileHandler("{}.log".format(log_name))
      file_handler.setLevel(logging.DEBUG)
      file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      file_handler.setFormatter(file_formatter)
      self.log.addHandler(file_handler)

    df = pd.read_csv(linksfile)
    df['link'] = df.i.map(str) + '_' + df.j.map(str) + '_' + df.k.map(str)
    df.set_index('link', inplace=True)

    self.df = df
    self.linksfile = os.path.splitext(linksfile)[0] # filename w/o extension

    # self.T = len(self.df)
    SR_stats = pd.read_csv('calvin/data/SR_stats.csv', index_col=0).to_dict()
    self.min_storage = SR_stats['min']
    self.max_storage = SR_stats['max']

    if ic:
      self.apply_ic(ic)

    # a few network fixes to make things work
    self.add_ag_region_sinks()
    self.fix_hydropower_lbs()

    self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(df.i,df.j,df.k))
    self.networkcheck() # make sure things aren't broken
    

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
    available storage. Needed for limited foresight (annual) optimization.

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
    Hack to get rid of surplus water at no cost from agricultural regions.
    Called internally when model is initialized.

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
      self.df = self.df._append(links.drop_duplicates())


  def fix_hydropower_lbs(self):
    """
    Hack to fix lower bound constraints on piecewise hydropower links.
    Storage piecewise links > 0 should have 0.0 lower bound, and
    the k=0 pieces should always have lb = dead pool.

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
    
    :param debug_mode: (boolean) Whether to run in debug mode.
      Use when there may be infeasibilities in the network.
    :param debug_cost: When in debug mode, assign this cost ($/AF) to flow on debug links.
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

    self.log.info('Creating Pyomo Model (debug=%s)' % debug_mode)

    model = ConcreteModel()

    model.N = Set(initialize=self.nodes)
    model.k = Set(initialize=range(15))
    model.A = Set(within=model.N*model.N*model.k, 
                  initialize=self.links, ordered=True)
    model.source = Param(initialize='SOURCE', within=Any)
    model.sink = Param(initialize='SINK', within=Any)

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
    model.c = Param(model.A, initialize=init_params('cost'))

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

    from pyomo.opt import SolverFactory
    opt = SolverFactory(solver)

    if nproc > 1 and solver != 'glpk':
      opt.options['threads'] = nproc
    
    if debug_mode:
      run_again = True
      i = 0
      vol_total = 0

      while run_again and i < maxiter:
        self.log.info('-----Solving Pyomo Model (debug=%s)' % debug_mode)
        self.results = opt.solve(self.model)
        self.log.info('Finished. Fixing debug flows...')
        run_again,vol = self.fix_debug_flows()
        i += 1
        vol_total += vol

      if run_again:
        self.log.info(('Warning: Debug mode maximum iterations reached.'
               ' Will still try to solve without debug mode.'))
      else:
        self.log.info('All debug flows eliminated (iter=%d, vol=%0.2f)' % (i,vol_total))

    else:
      self.log.info('-----Solving Pyomo Model (debug=%s)' % debug_mode)
      self.results = opt.solve(self.model, tee=False)

      if self.results.solver.termination_condition == TerminationCondition.optimal:
        self.log.info('Optimal Solution Found (debug=%s).' % debug_mode)
        self.model.solutions.load_from(self.results)
      else:
        raise RuntimeError('Problem Infeasible. Run again starting from debug mode.')


  def fix_debug_flows(self, tol=1e-7):
    """
    Find infeasible constraints where debug flows occur.
    Fix them by either raising the UB, or lowering the LB.
    
    :param tol: (float) Tolerance to identify nonzero debug flows
    :returns run_again: (boolean) whether debug mode needs to run again
    :returns vol: (float) total volume of constraint changes
      also modifies the model object.
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
            self.log.info('%s UB raised by %0.2f (%0.2f%%)' % (l[0]+'_'+l[1], v, v*100/iv))
            df.loc['_'.join(str(x) for x in l[0:3]), 'upper_bound'] = model.u[s2].value

        # if we need to bring in extra water
        # this is a much more common problem
        # want to avoid reducing carryover requirements. look downstream instead.
        max_depth = 10

        if 'DBUGSRC' in dbl[0]:
          vol_to_reduce = max(model.X[s].value*1.2, 0.5)
          self.log.info('Volume to reduce: %.2e' % vol_to_reduce)

          children = [dbl[1]]
          for i in range(max_depth):
            children += df[df.i.isin(children)
                           & ~ df.j.str.contains('DBUGSNK')].j.tolist()
          children = set(children)
          reducelinks = (df[df.i.isin(children)
                           & (df.lower_bound > 0)]
                         .sort_values(by='lower_bound', ascending=False).values)

          if reducelinks.size == 0:
            raise RuntimeError(('Not possible to reduce LB on links'
                                ' with origin %s by volume %0.2f' % 
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
              self.log.info('%s LB reduced by %.2e (%0.2f%%). Dual=%.2e' % (l[0]+'_'+l[1], v, v*100/iv, dl))
              df.loc['_'.join(str(x) for x in l[0:3]), 'lower_bound'] = model.l[s2].value
              
            if vol_to_reduce == 0:
              break

          if vol_to_reduce > 0:
            self.log.info('Debug -> %s: could not reduce full amount (%.2e left)' % (dbl[1],vol_to_reduce))

    self.df, self.model = df, model
    return run_again, vol_total
