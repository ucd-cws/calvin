import json
import csv

def save_dict_as_csv(data, filename):
  node_keys = sorted(data.keys())
  time_keys = sorted(data[node_keys[0]].keys()) # add key=int for integer timesteps

  header = ['time'] + node_keys
  writer = csv.writer(open(filename, 'w'))
  writer.writerow(header)

  with open('REGIONS.csv','r') as f:
    reader=csv.reader(f)
    REGIONS = list(reader)


  region=['region']

  for i in range(1,len(header)):
    check=0;
    for j in range(len(REGIONS)):
      if header[i] == REGIONS[j][0]:
        region.append(REGIONS[j][1])
        check=1;
    if check == 0:
      region.append('Unknown')


  writer.writerow(region)


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
      if v is not None and (D[k1][k2] is None or v > D[k1][k2]):
        D[k1][k2] = v

# start with four empty dicts -- this is
# what we want to output (in separate files):
# flows (F), storages (S), and duals (D)
F,S,E,D_up,D_lo,D_node = {}, {}, {}, {}, {}, {}
ShortAgVol,ShortAgCost,ShortUrbVol,ShortUrbCost={},{},{},{}


# load network links
with open('links.csv', 'rU') as f:
  reader = csv.reader(f)
  network = list(reader)

# load network nodes
with open('nodes.csv', 'rU') as f:
  reader = csv.reader(f)
  network_nodes = list(reader)

#load urban and ag nodes
with open('agnodes_region_102016.csv', 'r') as f:
  reader = csv.reader(f)
  agnodes = list(reader)
  # agg=list()
  # for i in range(len(agnodes)):
  #   agg.append('\n'.join(agnodes[i]))
  # agnodes=agg


with open('urbannodes_region_102016.csv', 'r') as f:
  reader = csv.reader(f)
  urbannodes = list(reader)
  # urbb=list()
  # for i in range(len(urbannodes)):
  #   urbb.append('\n'.join(urbannodes[i]))
  # urbannodes=urbb


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
    if is_storage_node == 1:
      amplitude = float(link[4])
  elif '.' in link[0] and link[1] == 'FINAL': # End-of-period storage for reservoirs
    n1,t1 = link[0].split('.')
    is_storage_node = True
    amplitude = 1
  else:
    continue


  # fix zeros in pyomo output  
  v = dict_get(flows, 'X[%s]' % s, 'Value', default = 0.0)
  d1 = dict_get(constraints, 'limit_upper[%s]' % s, 'Dual', default = None)
  d2 = dict_get(constraints, 'limit_lower[%s]' % s, 'Dual', default = None)


  # sum over piecewise components
  if is_storage_node:
    key = n1
    EVAP=(1-amplitude)*float(v)/amplitude
    dict_insert(S, key, t1, v, 'sum')
    dict_insert(E, key, t1, EVAP, 'sum')
  else:
    key = n1 + '-' + n2
    dict_insert(F, key, t1, v, 'sum')
    # #Check for urban or ag demands
    TOL=1e-6;
    for aglink in agnodes:
      if key in aglink[0]:
        if (float(link[6])-float(v))>TOL:
          dict_insert(ShortAgVol, key, t1, float(link[6])-float(v), 'sum')
          dict_insert(ShortAgCost, key, t1, float(link[3])*(float(link[6])-float(v)), 'sum')
        else:
          dict_insert(ShortAgVol, key, t1, 0, 'sum')
          dict_insert(ShortAgCost, key, t1, 0, 'sum')
    for urblink in urbannodes:
      if key in urblink[0]:
        if (float(link[6])-float(v))>TOL:
          dict_insert(ShortUrbVol, key, t1, float(link[6])-float(v), 'sum')
          dict_insert(ShortUrbCost, key, t1, float(link[3])*(float(link[6])-float(v)), 'sum')
        else:
          dict_insert(ShortUrbVol, key, t1, 0, 'sum')
          dict_insert(ShortUrbCost, key, t1, 0, 'sum')

  # open question: what to do about duals on pumping links? Is this handled?
  dict_insert(D_up, key, t1, d1, 'max')
  dict_insert(D_lo, key, t1, d2, 'max')


# store dual values nodes in a dictionary
for node in network_nodes:
  if '.' in node[0]:
    k = 'flow['+str(node[0])+']'
    n3,t3 = node[0].split('.')
    d3 = dict_get(constraints, k, 'Dual', default= None)
    dict_insert(D_node, n3, t3, d3, 'max')

# write the output files
things_to_save = [(F, 'flow'), (S, 'storage'), (D_up, 'dual_upper'), (D_lo, 'dual_lower'), (D_node, 'dual_node'),(E,'evaporation'),(ShortAgVol,'shortagvol'),(ShortAgCost,'shortagcost'),(ShortUrbCost,'shorturbcost'),(ShortUrbVol,'shorturbvol')]

for data,name in things_to_save:
  save_dict_as_csv(data, name + '.csv')

print('time-series successfully stored in csv files')