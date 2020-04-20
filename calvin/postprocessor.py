import os, json
import csv
import pandas as pd
import numpy as np


def save_dict_as_csv(data, filename, mode='w'):
    """
    Given nested dict `data`, write to CSV file
    where rows are timesteps and columns are links/nodes as appropriate.
    
    :param data: (dict) Nested dictionary of results data
    :param filename: (string) Output CSV filename
    :param mode: (string) file writing mode, 'w' for write, 'a' for append
        in particular the annual optimization uses 'a' to append results each year.
    :returns: nothing, but writes the output CSV file.
    """
    node_keys = sorted(data.keys())
    time_keys = sorted(data[node_keys[0]].keys()) # add key=int for integer timesteps

    writer = csv.writer(open(filename, mode))

    if mode == 'w':
        header = ['date'] + node_keys
        writer.writerow(header)

    for t in time_keys:
        row = [t]
        for k in node_keys:
            if t in data[k] and data[k][t] is not None:
                row.append(data[k][t])
            else:
                row.append(0.0)
        writer.writerow(row)


def dict_get(D, k1, k2, default = 0.0):
    """
    Custom retrieval from nested dictionary.
    Get D[k1][k2] if it exists, otherwise default.
    """
    if k1 in D and k2 in D[k1]:
        return D[k1][k2]
    else:
        return default


def dict_insert(D, k1, k2, v, collision_rule = None):
    """
    Custom insertion into nested dictionary.
    Assign D[k1][k2] = v if those keys do not exist yet.
    If the keys do exist, follow instructions for collision_rule
    """
    if k1 not in D:
        D[k1] = {k2: v}
    elif k2 not in D[k1]:
        D[k1][k2] = v
    else:
        if collision_rule == 'sum':
            D[k1][k2] += v
        # elif collision_rule == 'max':
        #   if v is not None and (D[k1][k2] is None or v > D[k1][k2]):
        #     D[k1][k2] = v
        elif collision_rule == 'first':
            pass # do nothing, we already have the first value
        elif collision_rule == 'last':
            D[k1][k2] = v # replace
        else:
            raise ValueError('Keys [%s][%s] already exist in dictionary' % (k1,k2))


def postprocess(df, model, resultdir=None, annual=False, year=None, regionalagg=False):
    """
    Postprocess model results into timeseries CSV files.

    :param df: (dataframe) network links data
    :param model: (CALVIN object) model object, post-optimization
    :param resultdir: (string) directory to place CSV file results
    :param annual: (boolean) whether to run annual optimization or not
    :returns EOP_storage: (dict) end-of-period storage, only when 
        running annual optimization. Otherwise returns nothing, but writes files.
    """
    import os
    # start with empty dicts -- this is
    # what we want to output (in separate files):
    # flows (F), storages (S), duals (D), evap (E), shortage vol (SV) and cost (SC)
    F,S,E,SV,SC = {}, {}, {}, {}, {}
    D_up,D_lo,D_node = {}, {}, {}
    eop,eop_storage,eop_cost = {}, {}, {}

    links = df.values
    nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
    demand_nodes = pd.read_csv(
        os.path.join(os.path.dirname(__file__),'data/california-network/demand_nodes.csv'), index_col = 0
    )

    for link in links:
        # get values from JSON results. If they don't exist, default is 0.0.
        # (sometimes pyomo does not include zero values in the output)
        s = tuple(link[0:3])
        ub = float(link[6])
        unit_cost = float(link[3])
        v = model.X[s].value if s in model.X else 0.0
        d1 = model.dual[model.limit_lower[s]] if s in model.limit_lower else 0.0
        d2 = model.dual[model.limit_upper[s]] if s in model.limit_upper else 0.0

        # now figure out what keys to save it in the dictionaries
        if '.' in link[0] and '.' in link[1]:
            n1,t1 = link[0].split('.')
            n2,t2 = link[1].split('.')
            is_storage_node = (n1 == n2)
            if is_storage_node:
                amplitude = float(link[4])
                is_eop = False
        elif '.' in link[0] and link[1] == 'FINAL': # End-of-period storage for reservoirs
            n1,t1 = link[0].split('.')
            is_storage_node = True
            is_eop = True
        else:
            continue

        # overwrite year if in lf_search_mode
        if not year==None and year>1922:
            t1 = t1.replace('1922',str(year)).replace('1921',str(year-1))
            # t1 = "{}-{}".format(year,t1[-5:]) 

        # sum over piecewise components
        if is_storage_node:
            key = n1
            if is_eop:
                if key in eop:
                    eop[key] += v 
                else:
                    eop[key] = v
                dict_insert(eop_storage, key, t1, v, 'sum')
                dict_insert(eop_cost, key, t1, -1*unit_cost*(ub-v), 'sum')
            else:
                evap = (1 - amplitude)*float(v)/amplitude
                dict_insert(S, key, t1, v, 'sum')
                dict_insert(E, key, t1, evap, 'sum')
        else:
            key = n1 + '-' + n2
            dict_insert(F, key, t1, v, 'sum')

            # Check for urban or ag demands
            if key in demand_nodes.index.values:
                if (ub - v) > 1e-6: # if there is a shortage
                    dict_insert(SV, key, t1, ub-v, 'sum')
                    dict_insert(SC, key, t1, -1*unit_cost*(ub-v), 'sum')
                else:
                    dict_insert(SV, key, t1, 0.0, 'sum')
                    dict_insert(SC, key, t1, 0.0, 'sum')

        # open question: what to do about duals on pumping links? Is this handled?
        dict_insert(D_up, key, t1, d1, 'last')
        dict_insert(D_lo, key, t1, d2, 'first')


    # get dual values for nodes (mass balance)
    for node in nodes:
        if '.' in node:
            n3,t3 = node.split('.')
            d3 = model.dual[model.flow[node]] if node in model.flow else 0.0
            dict_insert(D_node, n3, t3, d3)

    # write the output files
    import datetime, os

    if not resultdir:
        if annual:
            raise RuntimeError('resultdir must be specified for annual run')
        resultdir = 'results-' + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')

    mode = 'a' if annual and os.path.isdir(resultdir) else 'w'
    if not os.path.isdir(resultdir):
        os.makedirs(resultdir)

    things_to_save = [(F, 'flow'), (S, 'storage'), (D_up, 'dual_upper'), 
                                        (D_lo, 'dual_lower'), (D_node, 'dual_node'),
                                        (E,'evaporation'), (SV,'shortage_volume'),
                                        (SC,'shortage_cost'), (eop_storage,'eop_storage'), (eop_cost,'eop_cost')]

    for data,name in things_to_save:
        save_dict_as_csv(data, resultdir + '/' + name + '.csv', mode)

    if annual:
        return eop
    elif regionalagg:
        aggregate_regions(resultdir)


def aggregate_regions(fp):
    """
    Read the results CSV files and aggregate results by region (optional).

    :param fp: (string) directory where output files are written.
    :returns: nothing, but overwrites the results files with new
        columns added for regional aggregations.
    """
    import os
    # aggregate regions and supply portfolios
    # easier to do this with pandas by just reading the CSVs again
    sc = pd.read_csv(fp + '/shortage_cost.csv', index_col=0, parse_dates=True)
    sv = pd.read_csv(fp + '/shortage_volume.csv', index_col=0, parse_dates=True)
    flow = pd.read_csv(fp + '/flow.csv', index_col=0, parse_dates=True)
    demand_nodes = pd.read_csv(
        os.path.join(os.path.dirname(__file__),'data/california-network/demand_nodes.csv'), index_col = 0
    )
    portfolio = pd.read_csv(
        os.path.join(os.path.dirname(__file__),'data/california-network/portfolio.csv'), index_col = 0
    )

    for R in demand_nodes.region.unique():
        for t in demand_nodes.type.unique():
            ix = demand_nodes.index[(demand_nodes.region == R) & 
                                                            (demand_nodes.type == t)]
            sc['%s_%s' % (R,t)] = sc[ix].sum(axis=1)
            sv['%s_%s' % (R,t)] = sv[ix].sum(axis=1)

    for P in portfolio.region.unique():
        for k in portfolio.supplytype.unique():
            for t in portfolio.type.unique():
                ix = portfolio.index[(portfolio.region == P) & 
                                                         (portfolio.type ==t) & 
                                                         (portfolio.supplytype == k)]
                flow['%s_%s_%s' % (P,k,t)] = flow[ix].sum(axis=1)

    sc.to_csv(fp + '/shortage_cost.csv')
    sv.to_csv(fp + '/shortage_volume.csv')
    flow.to_csv(fp + '/flow.csv')


def load_cosvfpso_history(result_dir):
    """
    Load PSO iteration history csv's for post-processing and plotting
    """
    pos = pd.read_csv(os.path.join(result_dir, 'pso_position_history.csv'))
    pos.columns = ['iteration', 'particle', 'sr', 'param', 'position']

    cost = pd.read_csv(os.path.join(result_dir, 'pso_cost_history.csv'))
    pos = pos.merge(cost, on=['iteration', 'particle'])

    vel = pd.read_csv(os.path.join(result_dir, 'pso_velocity_history.csv'))
    vel.columns = ['iteration','particle','sr','param','velocity']
    pos = pos.merge(vel, on=['iteration', 'particle','sr','param'])

    pos['feasible'] = np.where(pos.cost > 0, False, True)
    bc = pos.loc[pos.feasible == True].groupby(['iteration'])['cost'].max().reset_index()
    bc['gbest'] = bc.cost
    pos = pos.merge(bc, how='left', on=['iteration', 'cost'])
    return pos


def get_best_cosvfpso_pminpmax(cosvfpso_history):
    """
    get the global best position per iteration
    """
    pminpmax = cosvfpso_history.loc[cosvfpso_history.gbest == cosvfpso_history.gbest.max()].position

    return list(pminpmax)

