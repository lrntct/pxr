# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
from scipy.special import gamma
import scipy.stats
import numpy as np


def KS_test(ds):
    """Perform the Kolmogorov–Smirnov test
    """
    ds = ds.chunk({'duration':-1})
    ds['analytic_prob'] = gumbel_cdf(ds['annual_max'], ds['location'], ds['scale'])
    ds['KS_D'] = xr.apply_ufunc(lambda x,y: np.max(np.abs(x-y)),
                                    ds['estim_prob'], ds['analytic_prob'],
                                    input_core_dims=[['year'], ['year']],
                                    vectorize=True,
                                    dask='parallelized',
                                    output_dtypes=[DTYPE]
                                    )
    # Dcrit
    len_dur = len(ds['duration'])
    alpha = ds['significance_level'] / 100.
    c = np.sqrt(-.5 * np.log(alpha))
    ds['KS_Dcrit'] = c * np.sqrt((len_dur+len_dur)/(len_dur*len_dur))
    return ds


def anderson_gumbel(x):
    try:
        return scipy.stats.anderson(x, dist='gumbel_r')[0]
    except RuntimeError:
        return np.nan


def anderson_darling(ds):
    """
    """
    # Get the critical values (depend only on the sample length)
    _, critical_values, significance_levels = scipy.stats.anderson(ds['year'], dist='gumbel_r')
    da_critical_values = xr.DataArray(critical_values, name='A2_crit',
                                      coords=[significance_levels],
                                      dims=['significance_level'])
    # Goodness of fit of the Gumbel distribution
    da_a2 = xr.apply_ufunc(anderson_gumbel,
                           ds['annual_max'],
                           input_core_dims=[['year']],
                           vectorize=True,
                           dask='parallelized',
                           output_dtypes=[DTYPE]
                           ).rename('A2')
    return da_critical_values, da_a2
