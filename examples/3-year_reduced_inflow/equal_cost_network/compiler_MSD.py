import pandas as pd
import csv

#use only networklinks.tsv to create list of nodes
df = pd.read_csv('links.csv',header=0)

val = 1 # new cost value that we will assign
for x,item in enumerate(df.cost): 
	if item < 2000000: # 2000000 is cost of debug flow, which we won't change
		df.set_value(x, 'cost', val)

coeff = 0.5
for y,item in enumerate(df.i): 
	if item[:6] == 'INFLOW': # reduce / increase inflows by coefficient amount
		df.set_value(y, 'lower_bound', coeff*df.lower_bound[y])
		df.set_value(y, 'upper_bound', coeff*df.upper_bound[y])

df.to_csv('links.tsv',sep='\t',index=False,header=False)
df_sorted = pd.unique(df[['i','j']].values.ravel())
nodes_test = df_sorted.tolist()

with open('nodes.tsv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter='\t')
    wr.writerow(nodes_test)

#import links and strip /n
links = open('links.tsv','r')
h = links.read()
linksedit = h.rstrip()

nodes = open('nodes.tsv','r')
g = nodes.read()
nodesedit = g.rstrip()

#write data.dat file
f = open('data.dat','w')

f.write('set N :=\n')
f.write(nodesedit + ';\n\n')

f.write('set k := 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;\n\n')

f.write('param source := SOURCE;\n')
f.write('param sink := SINK;\n\n')

f.write('param: A: c a l u :=\n')
f.write(linksedit + ';')




