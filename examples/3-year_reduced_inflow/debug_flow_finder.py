import csv

# load link flows
with open('flow.csv', 'rU') as f:
  reader = csv.reader(f)
  flows = list(reader)

nrows = len(flows[0])
ncols = len(flows)

# loop over columns (links)...
for j in range(1,nrows):
  l1,l2 = flows[0][j].split('-')

  # if this is a debug link ...
  if l1 == 'DBUGSRC' or l2 == 'DBUGSNK':
    ID = l1+'-'+l2

    # loop over the time series of flow values
    for i in range(1,ncols):

      # if any are nonzero, print them
      if float(flows[i][j]) > 0:
        print('Link: ' + ID + ', Time: ' + flows[i][0] + ', Flow: ' + flows[i][j])