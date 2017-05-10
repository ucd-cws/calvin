import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')

trial = list(range(1,11)) # number of trials: 10
solver = ['cplex', 'gurobi', 'cbc', 'glpk'] # solvers
data = ['1-year', '5-year', '10-year', '40-year', '82-year'] # run sizes (based on time period)
dec_var = [61670, 308138, 616223, 2464733, 5052647] # number of decision variables for each run (data) size

# create a matrix to store results
ncols = 9
nrows = len(trial)*len(solver)*len(data)
matrix = [[0] * ncols for i in range(nrows)] # create a matrix with zero values

# get results from text files
i = 0 # first run (total 200)
for x,d in enumerate(data):
	for s in solver:
		for t in trial:
			path = "benchmark_runs/" + d + "/" + s + "/" + "output_trial_" + str(t) + ".txt"
			f = open(path)
			next = f.readlines()
			matrix[i][0] = d
			matrix[i][1] = dec_var[x]
			matrix[i][2] = s
			matrix[i][3] = t
			for line in next:
				if line[-16:] == "Applying solver\n": # time from beginning to sending model to solver. pyomo sends the model to solver
					matrix[i][4] = float(line.split('[')[1].split(']')[0])
				if line[-19:] == "Processing results\n": # total time from benning to solver find a solution. pyomo gets results from solver
					matrix[i][5] = float(line.split('[')[1].split(']')[0])
				if line[-15:] == "Pyomo Finished\n": # time for pyomo to finish the run. pyomo finishes the runs
					matrix[i][6] = float(line.split('[')[1].split(']')[0])
				if line[:20] == "      Function Value": # objective value
					matrix[i][7] = float(line.split(':')[1])
				if line[:9] == "Time used": # glpk solver time
					matrix[i][8] = float(line.split('Time used:')[1].split('secs')[0])
				if line[:24] == "Total time (CPU seconds)": # cbc solver time
					matrix[i][8] = float(line.split('Total time (CPU seconds):')[1].split('(Wallclock seconds):')[0])					
				if line[:13] == "Solution time": # cplex solver time
					matrix[i][8] = float(line.split('Solution time =')[1].split('sec.')[0])				
				if line[:9] == "Solved in": # gurobi solver time
					matrix[i][8] = float(line.split('iterations and')[1].split('seconds')[0])
			i += 1 # next run
# write results to .csv file
header = ["data size", "decision variables", "solver", "trial", "applying solver (sec)", "processing results (sec)", "pyomo finished (sec)", "objective (K$)", "solver reported time (s)"]
with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(matrix)

# linear regression (loglog scale)
# website: http://stackoverflow.com/questions/18760903/fit-a-curve-using-matplotlib-on-loglog-scale
def add_trendline(x,y,c,l): # y = e**(a*log(x)-b) => y = 1/(e**b)x**a
	logx = np.log(x)
	logy = np.log(y)
	coeffs = np.polyfit(logx,logy,deg=1)
	poly = np.poly1d(coeffs)
	yfit = lambda x: np.exp(poly(np.log(x)))
	plt.loglog(x,yfit(x), color = c, linestyle = 'dashed', alpha=0.5, linewidth = 1)
	return coeffs
# smome statistics: mean, meadian, standard deviation
def stats(x):
	m = np.mean(x)
	med = np.median(x)
	std = np.std(x)
	return m, med, std

###
c = 6 # 6 is pyomo finished, 8 is solver reported time
###

# return statistics of solver and total model runtime
statistics = [[],[],[],[],[]]
for d in dec_var:
	for s in solver:
		x = []
		for line in matrix:
			if s == line[2] and d == line[1]:
				x.append(line[c])
		statistics[0].append(d)
		statistics[1].append(s)
		statistics[2].append(np.around(stats(x)[0], decimals=2))
		statistics[3].append(np.around(stats(x)[1], decimals=2))
		statistics[4].append(np.around(stats(x)[2], decimals=2))
statistics = np.array(statistics).T
# write results to .csv
header_stats = ['decision variables', 'solver', 'mean (s)', 'median (s)', 'std. deviation (s)']
with open('stats_'+header[c]+'.csv', "wb") as f:
    writer = csv.writer(f)
    writer.writerow(header_stats)
    writer.writerows(statistics)

# plotting stuff
h = [1, 1*60, 1*60*60, 1*60*60*24, 1*60*60*24*7] # some time marks
ht = ['1 second', '1 minute', '1 hour', '1 day', '1 week']

color = ['g', 'c', 'm', 'y', 'b'] # colors
marker = ['o', 'v', 's', 'D', 'p'] # markers
cplex, gurobi, cbc ,glpk = [[],[]],[[],[]],[[],[]],[[],[]]
for line in matrix:
	if line[2] == 'cplex':
		cplex[0].append(line[1]),cplex[1].append(line[c])
	if line[2] == 'gurobi':
		gurobi[0].append(line[1]),gurobi[1].append(line[c])
	if line[2] == 'cbc':
		cbc[0].append(line[1]),cbc[1].append(line[c])
	if line[2] == 'glpk':
		glpk[0].append(line[1]),glpk[1].append(line[c])

# add trendlines and points
a = []
for i in range(2): a.append(add_trendline(cplex[0],cplex[1],color[0],solver[0])[i])
plt.scatter(cplex[0],cplex[1], label = solver[0] + ' : ' + '$y=e^{' + str(np.around(a[1], decimals=3)) + '}$' + '$x^{' + str(np.around(a[0], decimals=3)) + '}$', marker = marker[0], color = color[0], alpha=0.5)

a = []
for i in range(2): a.append(add_trendline(gurobi[0],gurobi[1],color[1],solver[1])[i])
plt.scatter(gurobi[0],gurobi[1], label = solver[1] + ' : ' + '$y=e^{' + str(np.around(a[1], decimals=3)) + '}$' + '$x^{' + str(np.around(a[0], decimals=3)) + '}$', marker = marker[1], color = color[1], alpha=0.5)

a = []
for i in range(2): a.append(add_trendline(cbc[0],cbc[1],color[2],solver[2])[i])
plt.scatter(cbc[0],cbc[1], label = solver[2] + ' : ' + '$y=e^{' + str(np.around(a[1], decimals=3)) + '}$' + '$x^{' + str(np.around(a[0], decimals=3)) + '}$', marker = marker[2], color = color[2], alpha=0.5)

a = []
for i in range(2): a.append(add_trendline(glpk[0],glpk[1],color[3],solver[3])[i])
plt.scatter(glpk[0],glpk[1], label = solver[3] + ' : ' + '$y=e^{' + str(np.around(a[1], decimals=3)) + '}$' + '$x^{' + str(np.around(a[0], decimals=3)) + '}$', marker = marker[3], color = color[3], alpha=0.5)

plt.loglog()
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
frame = plt.legend(frameon=True, fontsize = 12, loc = 2).get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('gray')
plt.axes().xaxis.grid(False)
plt.axes().yaxis.grid(True, which='minor')
plt.xlim([10**4,10**7])
plt.ylim([10**0,10**6])
plt.ylabel('time (second)', fontsize = 14)
plt.xlabel('number of decision variables', fontsize = 14)
plt.title(header[c], fontsize = 14)
for hl in range(len(h)): plt.axhline(y=h[hl], linewidth=1, color='gray', linestyle = 'dashed', alpha = 0.65)
for hl in range(len(h)): plt.text(1.03*10**7,h[hl], ht[hl], color = 'gray', rotation = 'horizontal',fontsize = 10)
for j in range(len(dec_var)): plt.text(dec_var[j]-10*10**(3.2+0.4*j),0.55, str(dec_var[j]), color = 'gray', rotation = 'horizontal',fontsize = 10)
plt.savefig('variable_'+str(c)+'.pdf', transparent=True)
plt.show()

# plotting stuff / group by solvers
for k,item in enumerate(dec_var):
	cplex, gurobi, cbc ,glpk = [],[],[],[]
	for line in matrix:
		if line[1] == item:
			if line[2] == 'cplex':
				cplex.append(line[c])
			if line[2] == 'gurobi':
				gurobi.append(line[c])
			if line[2] == 'cbc':
				cbc.append(line[c])
			if line[2] == 'glpk':
				glpk.append(line[c])
	y = [np.array(cplex),np.array(gurobi),np.array(cbc),np.array(glpk)]
	x = [1,2,3,4]
	for xe, ye in zip(x, y):
	    plt.plot([xe] * len(ye), ye, color=color[k], marker=marker[k], alpha=0.5, linewidth=3, markersize=10)
for i in range(len(dec_var)): plt.plot(2, 0, label = dec_var[i], marker = marker[i], color = color[i],alpha=0.5, linewidth=3, markersize=10)
frame = plt.legend(frameon=True, fontsize = 12, loc = 9).get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('gray')
plt.xticks([1, 2, 3, 4])
plt.axes().set_xticklabels(solver, fontsize = 12)
plt.yticks(fontsize = 12)
plt.axes().xaxis.grid(False)
plt.axes().yaxis.grid(True, which='minor')
plt.yscale('log')
plt.ylim([10**0,10**6])
plt.ylabel('time (second)', fontsize = 14)
plt.xlabel('solvers', fontsize = 14)
plt.title(header[c], fontsize = 14)
for hl in range(len(h)): plt.axhline(y=h[hl], linewidth=1, color='gray', linestyle = 'dashed', alpha = 0.65)
for hl in range(len(h)): plt.text(4.16,h[hl], ht[hl], color = 'gray', rotation = 'horizontal',fontsize = 10)
plt.savefig('solver_'+str(c)+'.pdf', transparent=True)
plt.show()