**New Debug Mode Examples**

Two examples runs are created to test the new debug mode. The new debug mode resets the network cost of all `links` (arcs) to `1 $/AF`, except for `SOURCE` and `SINK`, which have a unit cost of `2000000 $/AF`. Resetting costs allows us to better idetify mass balance problems in the network. One problem with full cost network is that we cannot assign debug flows, which shows mass balance problems, to constraints. Since the objective in `PyVIN` is to minimize overall cost, it would put the debug where it is most beneficial. In this case, it is least-cost. Although, linear programming and cost minimization seem to be a problem to identify mass balance violations, it could also help us find exact locations if we could change unit cost of links. Assigning `1 $/AF` to all but `SOURCE, SINK` links allows us to put debug flows in exact location and timing. If debug flow is `DBUGSRC`, the problematic constraint will be downstream, if `DBGSINK`, it will be upstream of debug flow.

*3-year_reduced_inflow*

This example is in debug mode with full network cost. Only change is that inflows, including rim inflows and groundwater, set to `50%`

*3-year_reduced_inflow/equal_cost_network*

This example is also in debug mode. However, network costs for all links are equal to $1 /AF.  
