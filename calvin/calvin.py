from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy as np
import pandas as pd

class Network():

  def __init__(self, linksfile):
    df = pd.read_csv(linksfile)
    df['link'] = df.i.map(str) + '_' + df.j.map(str) + '_' + df.k.map(str)
    df.set_index('link', inplace=True)
    self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(df.i,df.j,df.k))
    self.df = df

    self.T = len(self.df)
    self.networkcheck()


  def networkcheck(self):
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


  def create_pyomo_model(self):

    model = ConcreteModel()

    model.N = Set(initialize=self.nodes)
    model.k = Set(initialize=range(15))
    model.A = Set(within=model.N*model.N*model.k, 
                  initialize=self.links, ordered=True)
    model.source = Param(initialize='SOURCE')
    model.sink = Param(initialize='SINK')

    def init_params(p):
      return lambda model,i,j,k: self.df.loc[str(i)+'_'+str(j)+'_'+str(k)][p]

    model.u = Param(model.A, initialize=init_params('upper_bound'))
    model.l = Param(model.A, initialize=init_params('lower_bound'))
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


  def solve_pyomo_model(self, solver='glpk', nproc=1, verbose=False):
    from pyomo.opt import SolverFactory
    opt = SolverFactory(solver)

    if nproc > 1 and solver is not 'glpk':
      opt.options['threads'] = nproc
    
    self.results = opt.solve(self.model, tee=verbose)

    if self.results.solver.termination_condition == TerminationCondition.optimal:
      self.model.solutions.load_from(self.results)
    else:
      print('infeasible')
      # do something about it? max's code
