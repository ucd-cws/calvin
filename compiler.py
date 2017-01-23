import pandas as pd
import csv

#use only networklinks.tsv to create list of nodes
df = pd.read_csv('links.csv',header=0)
df.to_csv('linksupdated.tsv',sep='\t',index=False,header=False)
df_sorted = pd.unique(df[['i','j']].values.ravel())
nodes_test = df_sorted.tolist()

with open('nodesupdated.tsv', 'w') as myfile:
    wr = csv.writer(myfile, delimiter='\t')
    wr.writerow(nodes_test)

#import links and strip /n
links = open('linksupdated.tsv','r')
h = links.read()
linksedit = h.rstrip()

nodes = open('nodesupdated.tsv','r')
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




