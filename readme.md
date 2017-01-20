# PyVIN Network Flow Optimization Model

### PyVIN is still in active development and is not production-ready

This documentation explains the process for creating a new PyVIN run either using data from the HOBBES network or self-created data. MORE INFO ON PYVIN. To learn more on the formulation of PyVIN, [detailed information](https://github.com/msdogan/pyvin/blob/master/Documentation/pyvin_documentation.pdf) is located in the Documentation folder.

## Required Programs
- [Python 3](https://www.continuum.io/downloads)

  The Anaconda platform conveniently installs Python and useful libraries such as [SciPy](https://www.scipy.org/) and [NumPy](http://www.numpy.org/).
  
- [git](https://git-scm.com/downloads)

- [Pyomo](https://software.sandia.gov/downloads/pub/pyomo/PyomoInstallGuide.html)

  If using Anaconda, the following command line promptx will install Pyomo: 
  ```
  conda install -c cachemeorg pyomo
  conda install -c cachemeorg pyomo.extras
  ```
  
  To run PyVIN, a solver needs to be specified. Examples: GLPK, CBC, CPLEX. If using Anaconda, use the following prompt to install          GLPK:
  
  ``` 
  conda install pyomo pyomo.extras ipopt glpk --channel cachemeorg
  ```
  
    Note: If working in Jon Herman's Group on the HPC1 cluster at UC Davis, solvers are already installed on the cluster.
    
## Getting Started

### PyVIN Repository

A local version of the PyVIN repository is necessary to create the input files and run PyVIN. Pull the [PyVIN](https://github.com/msdogan/pyvin) repostiory to your local machine with the following command:

```
git clone https://github.com/msdogan/pyvin
```


### Creating data.dat

```data.dat``` contains all data necessary to run ```pyvin.py```. There are 2 ways to create ```data.dat`` as described in the following 2 sections.

#### HOBBES Network Matrix Export

To run PyVIN using CALVIN data, the HOBBES network allows users to download data to form ```data.dat```.

##### Create networklinks.tsv


First, the [Calvin Network Tools repository](https://github.com/ucd-cws/calvin-network-tools) provides documentation on downloading the repository for data extraction from HOBBES. Once set up, the repository can be queried for data extraction:

To create a model run (12 months) from HOBBES network: 
```
cnf matrix --format csv --start 2002-10 --stop 2003-10 --ts . --to links --max-ub 1000000 --verbose
```
The following are additional flags that can be added to the command above:

```
    -h, --help                      output usage information
    -f, --format <csv|tsv|dot|png>  Output Format, dot | png (graphviz required)
    -N, --no-header                 Supress Header
    -S, --ts <sep>                  Time step separator, default=@
    -F, --fs <sep>                  Field Separator, default=,
    -s, --start [YYYY-MM]           Specify start date for TimeSeries data
    -t, --stop [YYYY-MM]            Specify stop date for TimeSeries data
    -M, --max-ub <number>           Replace null upperbound with a big number.  Like 1000000
    -d, --debug [nodes]             Set debug nodes.  Either "ALL", "*" or comma seperated list of prmnames (no spaces)
    -d, --data [repo/path/data]     path to Calvin Network /data folder
    -T, --to <filename>             Send matrix to filename, default=STDOUT
    -O, --outnodes <filename>       Send list of nodes to filename, default=no output, can use STDOUT
    -p, --outbound-penalty <json>   Specify a penalty function for outbound boundary conditions. eg. [[10000,"-10%"],[0,0],[-10000,"10%"]]

```

To create a run with defined nodes only:

Example: SR_SHA and D5 between Oct 1983 to Sep 1984 in debug mode
```
cnf matrix --format csv --start 1983-10 --stop 1984-10 --ts . --to links --max-ub 10000000 nodes SR_SHA D5 --debug SR_SHA,D5 --verbose
```

The HOBBES Matrix export will create ```links.csv```.


##### Run compiler.py
  
The ```compiler.py``` script will use ```links.csv``` to create the ```data.dat``` file. ```linksupdated.tsv``` and ```nodesupdated.tsv``` are used for postprocessing the results. 


#### Template data.dat

The other method besides using the HOBBES Network Export method is creating your own data.dat file. A template is included [here](https://github.com/msdogan/pyvin/blob/master/examples/SR_CLE-D94/data_sr_cle-d94.dat) and explained in the [Pyvin documentation](https://github.com/msdogan/pyvin/blob/master/Documentation/pyvin_documentation.pdf).

In addition to ```data.dat```, ```linksupdated.tsv``` and ```nodesupdated.tsv``` are necessary to run the postprocessing scripts. These files contain data from ```data.dat``` but in a different format. Templates and instructions are provided [here](LINK).



## Run PyVIN

Once the input files are created, PyVIN can be run on a local machine or high performance computing cluster.  

### Local Machine
To run the optimization: 
```
pyomo solve --solver=glpk --solver-suffix=dual pyvin.py data.dat --json --report-timing --stream-solver
```
Note: ```pyvin.py``` and ```data.dat``` need to be located in the same same directory as where the command line prompt is being written.

### HPC1 Instructions

These instructions are specific to the [HPC1 Cluster](http://ssg.cs.ucdavis.edu/services/research/hpc1-cluster). In addition to the files above, a Slurm script (```filename.sh```) is necessary. Instructions on submitting PyVIN jobs to HPC1 are located [here](https://gist.github.com/jdherman/b48db79abb365363eb1fb8822417d996).

## Postprocesing Data

Once the optimization is complete, a ```results.json``` file is created. 

To process the results into a series of CSV files, the script ```postprocess.py``` requires the following files:

  - ```linksupdated.tsv```
  - ```nodesupdated.tsv```
  - ```demand_nodes.csv```
  - ```results.json```


The ```postprocess.py``` script creates the following CSV files:

  - ```flow.csv```
  - ```storage.csv```
  - ```dual_upper.csv```
  - ```dual_lower.csv```
  - ```dual_node.csv```
  - ```evaporation.csv```
  - ```shortage_cost.csv```
  - ```shortage_volume.csv```

The ```aggregate_regions.py``` script appends the region name and supply type to every link in the following CSVs created by ```postprocess.py```: ```flow.csv``` , ```demand_nodes.csv``` , ```shortage_volume.csv``` , ```shortage_cost.csv```.

### Example Data Visualization: Supply Portfoilio

The variety of data produced by PyVIN allows for many ways to visualizae the data. One example is the ```supply_portfolio_hatchedbarplot.py``` script, which plots the sum of flows by each CALVIN region, supply type, and urban/agricultural link type. The following figure is an example output of the script:

![PyVIN Supply Portfolio Figure](https://github.com/msdogan/pyvin/blob/data_dat_compiler/postprocessor/supply_portfolio.png)


  
