import json
import csv
import pandas as pd

def save_dict_as_csv(data, filename):
  node_keys = sorted(data.keys())
  time_keys = sorted(data[node_keys[0]].keys()) # add key=int for integer timesteps

  header = ['date'] + node_keys
  writer = csv.writer(open(filename, 'w'))
  writer.writerow(header)

  for t in time_keys:
    row = [t]
    for k in node_keys:
      if t in data[k] and data[k][t] is not None:
        row.append(data[k][t])
      else:
        row.append(0.0)
    writer.writerow(row)

def dict_get(D, k1, k2, default = 0.0):
  if k1 in D and k2 in D[k1]:
    return D[k1][k2]
  else:
    return default

def dict_insert(D, k1, k2, v, collision_rule = None):
  if k1 not in D:
    D[k1] = {k2: v}
  elif k2 not in D[k1]:
    D[k1][k2] = v
  else:
    if collision_rule == 'sum':
      D[k1][k2] += v
    # elif collision_rule == 'max':
    #   if v is not None and (D[k1][k2] is None or v > D[k1][k2]):
    #     D[k1][k2] = v
    elif collision_rule == 'first':
      pass # do nothing, we already have the first value
    elif collision_rule == 'last':
      D[k1][k2] = v # replace
    else:
      raise ValueError('Keys [%s][%s] already exist in dictionary' % (k1,k2))

# start with empty dicts -- this is
# what we want to output (in separate files):
# flows (F), storages (S), duals (D), evap (E), shortage vol (SV) and cost (SC)
F,S,E,SV,SC = {}, {}, {}, {}, {}
D_up,D_lo,D_node = {}, {}, {}

# load network links
with open('linksupdated.tsv', 'rU') as f:
  reader = csv.reader(f,delimiter='\t')
  network = list(reader)

# load network nodes
with open('nodesupdated.tsv', 'rU') as f:
  reader = csv.reader(f,delimiter='\t')
  network_nodes = list(reader)

# load list of demand nodes to find shortages/costs for
with open('demand_nodes.csv', 'r') as f:
  reader = csv.reader(f)
  demand_nodes = [row[0] for row in reader]

# results from Pyomo
with open('results.json', 'r') as f:
  results = json.load(f)

flows = results['Solution'][1]['Variable']
constraints = results['Solution'][1]['Constraint']

for link in network:
  s = ','.join(link[0:3])
  if '.' in link[0] and '.' in link[1]:
    n1,t1 = link[0].split('.')
    n2,t2 = link[1].split('.')
    is_storage_node = (n1 == n2)
    if is_storage_node:
      amplitude = float(link[4])
  elif '.' in link[0] and link[1] == 'FINAL': # End-of-period storage for reservoirs
    n1,t1 = link[0].split('.')
    is_storage_node = True
    amplitude = 1
  else:
    continue

  # get values from JSON results. If they don't exist, default is 0.0.
  # (sometimes pyomo does not include zero values in the output)
  v = dict_get(flows, 'X[%s]' % s, 'Value')
  d1 = dict_get(constraints, 'limit_upper[%s]' % s, 'Dual')
  d2 = dict_get(constraints, 'limit_lower[%s]' % s, 'Dual')

  # sum over piecewise components
  if is_storage_node:
    key = n1
    evap = (1 - amplitude)*float(v)/amplitude
    dict_insert(S, key, t1, v, 'sum')
    dict_insert(E, key, t1, evap, 'sum')
  else:
    key = n1 + '-' + n2
    dict_insert(F, key, t1, v, 'sum')

    # Check for urban or ag demands
    if key in demand_nodes:
      ub = float(link[6])
      unit_cost = float(link[3])
      if (ub - v) > 1e-6: # if there is a shortage
        dict_insert(SV, key, t1, ub-v, 'sum')
        dict_insert(SC, key, t1, -1*unit_cost*(ub-v), 'sum')
      else:
        dict_insert(SV, key, t1, 0.0, 'sum')
        dict_insert(SC, key, t1, 0.0, 'sum')

  # open question: what to do about duals on pumping links? Is this handled?
  dict_insert(D_up, key, t1, d1, 'last')
  dict_insert(D_lo, key, t1, d2, 'first')


# get dual values for nodes (mass balance)
for node in network_nodes:
  if '.' in node[0]:
    n3,t3 = node[0].split('.')
    d3 = dict_get(constraints,'flow[%s]' % node[0], 'Dual')
    dict_insert(D_node, n3, t3, d3)

# write the output files
things_to_save = [(F, 'flow'), (S, 'storage'), (D_up, 'dual_upper'), 
                  (D_lo, 'dual_lower'), (D_node, 'dual_node'),
                  (E,'evaporation'), (SV,'shortage_volume'),
                  (SC,'shortage_cost')]

for data,name in things_to_save:
  save_dict_as_csv(data, name + '.csv')
