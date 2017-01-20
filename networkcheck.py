import csv
import numpy as np

fp = './'

# load network links
with open(fp + 'links.csv', 'rU') as l:
  reader = csv.reader(l)
  links = list(reader)

# load network nodes
with open(fp + 'nodes.csv', 'rU') as n:
  reader = csv.reader(n)
  nodes = list(reader)

num_in = {n[0]: 0 for n in nodes} 
num_out = {n[0]: 0 for n in nodes}
lb_in = {n[0]: 0 for n in nodes} 
lb_out = {n[0]: 0 for n in nodes}
ub_in = {n[0]: 0 for n in nodes} 
ub_out = {n[0]: 0 for n in nodes}

nrows_l = len(links) # # of rows in links
nrows_n = len(nodes) # # of rows in nodes

# loop over links
for l in links[1:]:
  lb = float(l[5])
  ub = float(l[6])
  num_in[l[1]] += 1
  lb_in[l[1]] += lb
  ub_in[l[1]] += ub
  num_out[l[0]] += 1
  lb_out[l[0]] += lb
  ub_out[l[0]] += ub

  if lb > ub:
    print('lb > ub for link %s' % (l[0]+'-'+l[1]))



for n in nodes:
  if num_in[n[0]] == 0:
    print('no incoming link for ' + n[0])
  if num_out[n[0]] == 0:
    print('no outgoing link for ' + n[0])

  if ub_in[n[0]] < lb_out[n[0]]:
    print('ub_in < lb_out for %s (%d < %d)' % (n[0], ub_in[n[0]], lb_out[n[0]]))
  if lb_in[n[0]] > ub_out[n[0]]:
    print('lb_in > ub_out for %s (%d > %d)' % (n[0], lb_in[n[0]], ub_out[n[0]]))

# # Save results to a .csv file
# nodes = [node[0] for node in nodes]
# with open('networkcheck.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(["node", "# incoming", "# outgoing"]) # header
#     writer.writerows(zip(nodes, list(incoming), list(outgoing))) # results
