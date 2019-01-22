# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
from scipy.special import gamma
import scipy.stats
import numpy as np

import helper

"""Collection of functions related to extreme values probabilities
"""

DTYPE = 'float32'

def rank_ams(da_ams, chunks, dtype):
    """Rank the annual maxs in in ascending order
    """
    # Ranking does not work on dask array
    asc_rank = da_ams.load().rank(dim='year')
    # make sure that the resulting array is in the same order as the original
    ranks = asc_rank.rename('rank').astype(dtype).transpose(*da_ams.dims)
    # Merge arrays in a single dataset, set the dask chunks
    return xr.merge([da_ams, ranks]).chunk(chunks)


def ecdf_weibull(rank, n_obs):
    """Return the Weibull plotting position
    """
    return rank / (n_obs+1)


def ecdf_gringorten(rank, n_obs):
    """Return the Gringorten plotting position
    """
    return (rank - 0.44) / (n_obs + 0.12)


def ecdf_hosking(rank, n_obs):
    """Return the plotting position defined in:
    Hosking, J. R. M. (1990).
    L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics.
    Journal of the Royal Statistical Society: Series B (Methodological), 52(1), 105–124.
    https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """
    return (rank - 0.35) / n_obs


def gumbel_mom(ds):
    """Fit Gumbel using the method of moments
    (Maidment 1993, cited by Bougadis & Adamowki 2006)
    """
    magic_number1 = 0.45
    magic_number2 = 0.7797
    mean = ds['annual_max'].mean(dim='year')
    std = ds['annual_max'].std(dim='year')
    loc = (mean - (magic_number1 * std)).rename('location')
    scale = (magic_number2 * std).rename('scale')
    return loc, scale


def gumbel_iter_linear(ds):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    # linearize
    linearize = lambda a: (np.log(np.log(1/a))).astype(DTYPE)
    ecdf_linear = linearize(ds['ecdf_weibull'])
    # First fit. Keep only the two first returning DataArrays
    estim_slope, estim_intercept = helper.linregress(scipy.stats.linregress,
                                              ds['annual_max'],
                                              ecdf_linear, ['year'])[:2]
    # get provisional gumbel parameters
    loc_prov = -estim_intercept / estim_slope
    scale_prov = -1 / estim_slope
    # Analytic probability F(x) from Gumbel CDF
    analytic_prob = gumbel_cdf(ds['annual_max'], loc_prov, scale_prov)
    # Get the final location and scale parameters
    analytic_prob_linear = linearize(analytic_prob)
    analytic_slope, analytic_intercept = helper.linregress(scipy.stats.linregress,
                                                    ds['annual_max'],
                                                    analytic_prob_linear, ['year'])[:2]
    loc_final = (-analytic_intercept / analytic_slope).rename('location')
    scale_final = (-1 / analytic_slope).rename('scale')
    return loc_final, scale_final


def gumbel_cdf(x, loc, scale):
    z = (x - loc) / scale
    return np.e**(-np.e**-z)


def gumbel_mle_wrapper(ams):
    try:
        params = scipy.stats.gumbel_r.fit(ams)
    except RuntimeError:
        params = (np.nan, np.nan)
    return params[0], params[1]


def gumbel_mle_fit(ds, dtype=DTYPE):
    """Employ scipy stats to find the Gumbel coefficients
    """
    loc, scale = xr.apply_ufunc(gumbel_mle_wrapper,
                                ds['annual_max'],
                                input_core_dims=[['year']],
                                output_core_dims=[[], []],
                                vectorize=True,
                                dask='allowed',
                                output_dtypes=[dtype, dtype]
                                )
    return loc.rename('location'), scale.rename('scale')


def frechet_cdf(x, loc, scale, shape):
    z = (x - loc) / scale
    return np.e**(-z)**-shape


def frechet_mom(ds, shape=0.114):
    """Fit Fréchet (EV type II) using the method of moments and a fixed shape parameter.
    Koutsoyiannis, D. (2004).
    Statistics of extremes and estimation of extreme rainfall: II.
    Empirical investigation of long rainfall records.
    Hydrological Sciences Journal, 49(4).
    https://doi.org/10.1623/hysj.49.4.591.54424
    """
    c1 = math.sqrt(gamma(1-2*shape) - gamma(1-shape)**2)
    c3 = (gamma(1-shape)-1)/shape

    mean = ds['annual_max'].mean(dim='year')
    std = ds['annual_max'].std(dim='year')
    scale = (c1 * std).rename('scale')
    loc = ((mean/scale) - c3).rename('location')
    da_shape = xr.full_like(mean, shape).rename('shape')
    return loc, scale, da_shape

def gev_cdf_nonzero(x, loc, scale, shape):
    z = (x - loc) / scale
    return np.e**(-(1+shape*z)**-(1/shape))


def gev_cdf(x, loc, scale, shape):
    return xr.where(shape == 0,
                    gumbel_cdf(x, loc, scale),
                    gev_cdf_nonzero(x, loc, scale, shape))


def b_value(ds, order):
    """b values used for GEV fitting using the PWM
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    n_obs = ds['year'].count()
    # Hosking (1990) calls for a specific plotting position
    pr_sum = ((ds['ecdf_hosking']**order) * ds['annual_max']).sum(dim='year')
    return (1./n_obs) * pr_sum


def gev_pwm(ds, shape=None):
    """Fit the GEV using the Method of Probability-Weighted Moments
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    # b0 = ds['annual_max'].mean(dim='year')
    b0 = b_value(ds, 0)
    b1 = b_value(ds, 1)

    if shape:
        da_shape = xr.full_like(b1, shape)
    else:
        b2 = b_value(ds, 2)
        c = (2*b1 - b0) / (3*b2-b0) - math.log(2)/math.log(3)
        da_shape = 7.859 * c + 2.9554 * c**2
    scale = ((2*b1-b0) * da_shape) / (gamma(1+da_shape) * (1-2**-da_shape))
    location = b0 + scale * (gamma(1+da_shape)-1) / da_shape

    return location.rename('location'), scale.rename('scale'), da_shape.rename('shape')


def gev_mle_wrapper(ams):
    try:
        params = scipy.stats.genextreme.fit(ams)
    except RuntimeError:
        params = (np.nan, np.nan, np.nan)
    return params[1], params[2], params[0]


def gev_mle_fit(ds, dtype=DTYPE):
    """Employ scipy stats to find the GEV coefficients
    """
    loc, scale, shape = xr.apply_ufunc(gev_mle_wrapper,
                                ds['annual_max'],
                                input_core_dims=[['year']],
                                output_core_dims=[[], [], []],
                                vectorize=True,
                                dask='allowed',
                                output_dtypes=[dtype, dtype, dtype]
                                )
    return loc.rename('location'), scale.rename('scale'), shape.rename('shape')
