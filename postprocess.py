import json
import csv

def save_dict_as_csv(data, filename):
  node_keys = sorted(data.keys())
  time_keys = sorted(data[node_keys[0]].keys(), key=int) # add key=int for integer timesteps

  header = ['time'] + node_keys
  writer = csv.writer(open(filename, 'wb'))
  writer.writerow(header)
  
  for t in time_keys:
    row = [t]
    for k in node_keys:
      if t in data[k] and data[k][t] is not None:
        row.append(data[k][t])
      else:
        row.append(0.0)
    writer.writerow(row)

def dict_get(D, k1, k2, default=0.0):
  if k1 in D and k2 in D[k1]:
    return D[k1][k2]
  else:
    return default

def dict_insert(D, k1, k2, v, collision_rule):
  if k1 not in D:
    D[k1] = {k2: v}
  elif k2 not in D[k1]:
    D[k1][k2] = v
  else:
    if collision_rule == 'sum':
      D[k1][k2] += v
    elif collision_rule == 'max':
      if v is not None and v > D[k1][k2]:
        D[k1][k2] = v

# start with four empty dicts -- this is
# what we want to output (in separate files):
# flows (F), storages (S), and duals (D)
F,S,D_up,D_lo = {}, {}, {}, {}

# load network links
with open('links.csv', 'rU') as f:
  reader = csv.reader(f)
  network = list(reader)

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
  else:
    continue # do not include boundary conditions right now

  # fix zeros in pyomo output  
  v = dict_get(flows, 'X[%s]' % s, 'Value', default = 0.0)
  d1 = dict_get(constraints, 'limit_upper[%s]' % s, 'Dual', default = None)
  d2 = dict_get(constraints, 'limit_lower[%s]' % s, 'Dual', default = None)

  # sum over piecewise components
  if is_storage_node:
    key = n1
    dict_insert(S, key, t1, v, 'sum')
  else:
    key = n1 + '_' + n2
    dict_insert(F, key, t1, v, 'sum')

  # open question: what to do about duals on pumping links? Is this handled?
  dict_insert(D_up, key, t1, d1, 'max')
  dict_insert(D_lo, key, t1, d2, 'max')

# write the output files
things_to_save = [(F, 'flow'), (S, 'storage'), (D_up, 'dual_upper'), (D_lo, 'dual_lower')]

for data,name in things_to_save:
  save_dict_as_csv(data, name + '.csv')

   