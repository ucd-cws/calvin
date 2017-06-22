### Advanced readme for data export

Users interested in exporting a subset of the California network can create their own CSV files using these steps.

The [Calvin Network Tools repository](https://github.com/ucd-cws/calvin-network-tools) provides documentation for downloading the full dataset and querying for extraction. Follow those instructions (note that Node.js is required).

For example, to export a statewide 12-month network called `links.csv` from the HOBBES dataset: 
```bash
cnf matrix --format csv --start 2002-10 --stop 2003-10 --ts . \
--to links --max-ub 1000000000000 --verbose
```
(The `--max-ub` flag specifies an arbitrarily large number to use for links that do not have a specified upper bound. We have found that `1e12` usually works.)

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

To create a run with only a specific set of nodes, for example SR_SHA and D5 between Oct 1983 to Sep 1984 in debug mode:
```
cnf matrix --format csv --start 1983-10 --stop 1984-10 --ts . \
--to links --max-ub 1000000000000 nodes SR_SHA D5 --debug all --verbose
```

In both cases, the file `links.csv` will be created. This can be used as described in the main readme file.