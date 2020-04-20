import os, json
import numpy as np
import pandas as pd
import pwlf


def load_sr_dict(directory):
    """
    load reservoir dictionary
    """
    with open(os.path.join(directory, 'sr_dict.json')) as f:
        sr_dict = json.load(f)
    return sr_dict


def load_lf_inflows(directory, trace_column):
    """
    load inflows
    """
    inflows = pd.read_csv(os.path.join(directory, 'inflows.csv'),
                          index_col=0, parse_dates=True, infer_datetime_format=True)
    inflows = inflows.pivot(columns='j', values=trace_column).rename_axis(None, axis=1)

    return inflows


def cosvf_cosp(coeff, eops):
    """Determine penalty on end-of-period storage

    :param coeff: (array) a, b, and c quadratic cosvf coefficients
    :param eops: (array) end-of-period storage series

    :returns cosp: (float) total penalty on carryover storage [$]
    """
    cosp = coeff[0] * eops**2 + coeff[1] * eops + coeff[2]
    return cosp


def cosvf_fit(pmin, pmax, eop_min, eop_max):
    """Determine piecewise costs for cosvf

    :param pmin: (float) penalty representing maximum willingness
        to pay for an additional unit of storage that would encroach the
        rain-flood conservation pool (Pmin at eops=Kcs)
    :param pmax: (float) penalty representing maximum
        willingness to pay for an additional unit of storage (Pmax at eops=0)]
    :param eop_min: (float) end-of-period storage minimum bound [acre-feet]
    :param eop_max: (float) end-of-period storage upper bound (carryover
      capacity) [acre-feet]

    :returns x: (numpy.ndarray) storage values
             y: (numpy.ndarray) penalty amount per storage value

    """
    # determine COSVF coefficients based on pmin and pmax
    a = (pmin - pmax) / (2 * eop_max)
    b = pmax
    c = -1 * (eop_max * (pmin + pmax)) / 2
    coeff = [a, b, c]

    # build cosvf curve with penalties based on end-of-period storage series
    x = np.linspace(eop_min, eop_max, 500)
    y = cosvf_cosp(coeff, x)

    return x, y


def cosvf_pwlf(x, y, pw_n):
    """
    fit a n-part piecewise function to the cosvf

    :param x: (array) storage values
    :param y: (array) penalty values
    :param pw_n: (integer) number of piecewise links to fit to cosvf
    
    :returns sr_b: (list) storage link breakpoints
             sr_k: (list) storage link costs
    """
    sr_pwlf = pwlf.PiecewiseLinFit(x, y)
    b = sr_pwlf.fitfast(pw_n, 4)
    k = sr_pwlf.calc_slopes()

    # determine piecewise costs for cosvf
    sr_b, sr_k = [], []
    sr_b.append(b[1])
    sr_k.append(k[0])
    for i in range(1, len(b)-1):
        bound = b[i+1] - b[i]
        marg_cost = k[i]
        sr_b.append(bound)
        sr_k.append(marg_cost)

    return sr_b, sr_k


def build_trace_list(start, end, include_hist=True):
    """
    construct a list of synthetic trace names
    """
    trace_list = ['f' + str(x) for x in np.arange(start, end + 1)]
    if include_hist:
        trace_list.insert(0, 'hist')

    return trace_list
