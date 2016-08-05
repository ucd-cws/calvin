import csv
import numpy as np

# load network links
with open('networklinks.csv', 'rU') as l:
  reader = csv.reader(l)
  links = list(reader)

# load network nodes
with open('networknodes', 'rU') as n:
  reader = csv.reader(n)
  nodes = list(reader)

nrows_l = len(links) # # of rows in links
nrows_n = len(nodes) # # of rows in nodes

incoming = np.zeros(nrows_n) # number of incoming links
outgoing = np.zeros(nrows_n) # number of outgoing links

# check the numbers of incoming and outgoing links for each node
for i in range(0,nrows_n):
	count1, count2 = 0, 0
	for j in range(1,nrows_l): # start from the second row. First row is a header
		if nodes[i][0] == links[j][1]: # check for incoming links
			count1 += 1
		if nodes[i][0] == links[j][0]: # check for outgoing links
			count2 += 1
	incoming[i] = count1
	outgoing[i] = count2

	if count1 == 0:
		print('no incoming link for ' + str(nodes[i]))
	if count2 == 0:
		print('no outgoing link for ' + str(nodes[i]))

# Save results to a .csv file
nodes = [node[0] for node in nodes]
with open('networkcheck.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["node", "# incoming", "# outgoing"]) # header
    writer.writerows(zip(nodes, list(incoming), list(outgoing))) # results