#!/bin/bash

# to export 82-year network (perfect foresight)
cnf matrix --format csv --start 1921-10 --stop 2003-10 \
--ts . --to links82yr --max-ub 1000000000000 --debug all --verbose

# to export annual matrices in separate files (limited foresight)
# mkdir annual && cd annual

# for i in {1922..2003} 
# do 
# cnf matrix --format csv --start $((i-1))-10 --stop $i-10 \
# --ts . --to linksWY$i --max-ub 1000000000000 --debug all --verbose
# done
