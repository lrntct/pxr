# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
import numpy as np
import numba as nb
import scipy.stats
import bottleneck

import ev_fit


def KS_test(ecdf, cdf):
    """Retun the Kolmogorov–Smirnov test statistic
    """
    return np.abs(ecdf - cdf).max(dim='year')


def lilliefors_Dcrit(ds, chunks, shape, n_sample=10000):
    """Estimate the critical values of the KS test using statistical simulation.
    The critical value corresponds to the quantile of all the KS stats.

    See also:
    Wilks, D. S. (2011).
    Frequentist Statistical Inference.
    International Geophysics, 100, 133–186.
    https://doi.org/10.1016/B978-0-12-385022-5.00005-1
    """
    significance_levels = [0.01, 0.05, 0.1]
    n_obs = int(ds['n_obs'].max())  # more stringent for longer record length
    ks_d = lilliefors_Dcrit_gev(n_obs, significance_levels, shape, n_sample=10000)
    ds['Dcrit'] = xr.DataArray(ks_d, name='Dcrit',
                               coords=[significance_levels],
                               dims=['significance_level'])
    return ds.chunk(chunks)


def lilliefors_Dcrit_gev(n_obs, significance_levels, shape, n_sample=10000):
    """Estimate the critical values of the KS test using statistical simulation.
    The critical value corresponds to the quantile of all the KS stats.
    """
    q_levels = [1-i for i in significance_levels]
    D_list = []
    for i in range(n_sample):
        ams_sim = scipy.stats.genextreme.rvs(c=shape, size=n_obs)
        rank = bottleneck.rankdata(ams_sim)
        ecdf = rank / n_obs
        loc, scale, shape = ev_fit.gev_pwm(ams_sim, ecdf, n_obs,
                                           ax_year=0, shape=np.full((1), shape))
        cdf = ev_fit.gev_cdf(ams_sim, loc, scale, shape)
        D = np.abs(ecdf-cdf).max()
        D_list.append(D)
    return np.quantile(D_list, q_levels)
