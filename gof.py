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
import ev_quantiles
import helper


def KS_test(ecdf, cdf):
    """Retun the Kolmogorov–Smirnov test statistic
    """
    return np.abs(ecdf - cdf).max(dim='year')


def lilliefors_Dcrit(ds, chunks, shape, n_sample=10000):
    """Estimate the critical values of the KS test using statistical simulation.
    """
    significance_levels = [0.05, 0.1]
    n_obs = int(ds['n_obs'].max())  # more stringent for longer record length
    ks_d = lilliefors_Dcrit_gev(n_obs, significance_levels, shape, n_sample=10000)
    ds['Dcrit'] = xr.DataArray(ks_d, name='Dcrit',
                               coords=[significance_levels],
                               dims=['significance_level'])
    return ds.chunk(chunks)


def lilliefors_Dcrit_gev(n_obs, significance_levels, shape, n_sample=10000):
    """Estimate the critical values of the KS test using statistical simulation.
    See also:
    Wilks, D. S. (2011).
    Frequentist Statistical Inference.
    International Geophysics, 100, 133–186.
    https://doi.org/10.1016/B978-0-12-385022-5.00005-1
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


def filliben_test(ds):
    """Filliben normality test for GEV.
    Also called probability plot correlation coefficient (PPCC).
    Wilks, D. S. (2011).
    Frequentist Statistical Inference.
    International Geophysics, 100, 133–186.
    https://doi.org/10.1016/B978-0-12-385022-5.00005-1
    """
    pp_cunnane = ev_fit.pp_cunnane(ds['rank'], ds['n_obs'])
    T = 1 / (1 - pp_cunnane)
    loc = ds['gev'].sel(ci='estimate', ev_param='location')
    scale = ds['gev'].sel(ci='estimate', ev_param='scale')
    shape = ds['gev'].sel(ci='estimate', ev_param='shape')
    quantile_estimate = ev_quantiles.gev_quantile(T, loc, scale, shape)
    r = xr.apply_ufunc(
        helper.pearson_r,
        ds['annual_max'], quantile_estimate,
        kwargs={'axis': -1},
        vectorize=True,
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[]],
        output_dtypes=[np.float32],
        dask='parallelized',
        )
    ds['pp_cunnane'] = pp_cunnane
    ds['T_Cunnane'] = T
    ds['quantile_estimate'] = quantile_estimate
    ds['filliben_stat'] = r
    return ds


def filliben_crit(shape, n_obs):
    """Critical values for the Filliben Q-Q regression test for GEV.
    Heo, J.-H., Kho, Y. W., Shin, H., Kim, S., & Kim, T. (2008).
    Regression equations of probability plot correlation coefficient
    test statistics from several probability distributions.
    Journal of Hydrology, 355(1–4), 1–15.
    https://doi.org/10.1016/J.JHYDROL.2008.01.027
    """
    if shape > 0.25 or shape < -0.20:
        raise NotImplementedError
    q005_1 = 1.527 - 0.7656*shape + 2.228*shape**2 - 3.824*shape**3
    q005_2 = n_obs**(0.1986 + 0.3858*shape - 0.5985*shape**2)
    q_005 = 1 - np.e**-(q005_1 * q005_2)
    q_01_1 = 1.695 - 0.5205*shape + 1.229*shape**2 - 2.809*shape**3
    q_01_2 = n_obs**(0.1912 + 0.2838*shape - 0.3765*shape**2)
    q_01 = 1 - np.e**-(q_01_1 * q_01_2)
    return np.array([q_005, q_01])

