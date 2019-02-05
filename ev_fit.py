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

# Euler-Mascheroni constant
EM = 0.577215664901532860606512090082


def rank_ams(da_ams, dtype):
    """Rank the annual maxs in in ascending order
    """
    # Ranking does not work on dask array
    asc_rank = da_ams.load().rank(dim='year')
    # make sure that the resulting array is in the same order as the original
    ranks = asc_rank.rename('rank').astype(dtype).transpose(*da_ams.dims)
    return ranks


def plotting_position(rank, n_obs, a, b):
    """General plotting position
    """
    return (rank - a) / (n_obs + b)


def ecdf_weibull(rank, n_obs):
    """Return the Weibull plotting position
    Recommended by:
    Makkonen, Lasse. 2006.
    “Plotting Positions in Extreme Value Analysis.”
    Journal of Applied Meteorology and Climatology 45 (2): 334–40.
    https://doi.org/10.1175/JAM2349.1.
    """
    return plotting_position(rank, n_obs, 0, 1)


def ecdf_gringorten(rank, n_obs):
    """Return the Gringorten plotting position
    """
    return plotting_position(rank, n_obs, 0.44, .12)


def ecdf_hosking(rank, n_obs):
    """Return the plotting position defined in:
    Hosking, J. R. M. (1990).
    L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics.
    Journal of the Royal Statistical Society: Series B (Methodological), 52(1), 105–124.
    https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """
    return plotting_position(rank, n_obs, 0.35, 0)


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
    return loc, scale, 0


def gumbel_iter_linear(ds):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    # linearize
    linearize = lambda a: (np.log(np.log(1/a))).astype(DTYPE)
    reduced_variable = linearize(ds['ecdf_gringorten'])
    # First fit. Keep only the two first returning DataArrays
    estim_slope, estim_intercept = helper.linregress(scipy.stats.linregress,
                                              ds['annual_max'],
                                              reduced_variable, ['year'])[:2]
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
    """
    """
    z = (x - loc) / scale
    return np.e**(-z)**shape


def frechet_mom(ds, shape=0.114):
    """Fit Fréchet (EV type II) using the method of moments and a fixed shape parameter.
    EV type II when shape>0.
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


def gumbel_scale(l2):
    return l2 / math.log(2)


def gumbel_loc(l1, scale):
    return l1 - EM * scale


def gev_scale(l2, shape):
    """
    EV type II when shape<0.
    Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    return (l2 * shape) / (gamma(1 + shape) * (1 - 2**-shape))


def gev_loc(l1, scale, shape):
    """
    EV type II when shape<0.
    Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    return l1 - scale * ((1 - gamma(1 + shape)) / shape)


def gev_shape(b0, b1, b2):
    """
    EV type II when shape<0.
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    c = (2*b1 - b0) / (3*b2 - b0) - math.log(2) / math.log(3)
    return 7.859 * c + 2.9554 * c**2


def gumbel_pwm(ds):
    """
    Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    b0 = b_value(ds, 0)
    b1 = b_value(ds, 1)
    l1 = b0
    l2 = 2 * b1 - b0
    scale = gumbel_scale(l2)
    loc = gumbel_loc(l1, scale)
    # return shape = 0 for consistency
    shape = xr.full_like(scale, 0).rename('shape')
    return loc.rename('location'), scale.rename('scale'), shape


def frechet_pwm(ds):
    """According to [1], the parameters are estimated the
    same way as the GEV, the shape is just capped to zero.
    Fit according to [2].

    [1] Koutsoyiannis, D. (2004).
    Statistics of extremes and estimation of extreme rainfall: II.
    Empirical investigation of long rainfall records.
    Hydrological Sciences Journal, 49(4).
    https://doi.org/10.1623/hysj.49.4.591.54424
    [2] Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    b0 = b_value(ds, 0)
    b1 = b_value(ds, 1)
    b2 = b_value(ds, 2)
    l1 = b0
    l2 = 2 * b1 - b0

    raw_shape = gev_shape(b0, b1, b2)
    # Shape must be negative for Fréchet
    shape = xr.ufuncs.fmin(raw_shape, 0)
    # shape = xr.where(raw_shape >= 0,
    #                  xr.full_like(raw_shape, 0),
    #                  raw_shape)
    # print(shape)
    scale = xr.where(shape == 0,
                     gumbel_scale(l2),
                     gev_scale(l2, shape))
    loc = xr.where(shape == 0,
                   gumbel_loc(l1, scale),
                   gev_loc(l1, scale, shape))

    return loc.rename('location'), scale.rename('scale'), shape.rename('shape')


def gev_pwm(ds, shape=None):
    """Fit the GEV using the Method of Probability-Weighted Moments.
    EV type II when shape<0.

    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    b0 = b_value(ds, 0)
    b1 = b_value(ds, 1)
    l1 = b0
    l2 = 2 * b1 - b0

    if shape:
        da_shape = xr.full_like(b1, shape)
    else:
        b2 = b_value(ds, 2)
        da_shape = gev_shape(b0, b1, b2)
    scale = xr.where(da_shape == 0,
                     gumbel_scale(l2),
                     gev_scale(l2, da_shape))
    loc = xr.where(da_shape == 0,
                   gumbel_loc(l1, scale),
                   gev_loc(l1, scale, da_shape))
    return loc.rename('location'), scale.rename('scale'), da_shape.rename('shape')


def gev_cdf_nonzero(x, loc, scale, shape):
    """Consider an EV type II if shape<0
    """
    z = (x - loc) / scale
    return np.e ** (-(1-shape*z)**(1/shape))


def gev_cdf(x, loc, scale, shape):
    return xr.where(shape == 0,
                    gumbel_cdf(x, loc, scale),
                    gev_cdf_nonzero(x, loc, scale, shape))


def b0(ams, axis, n_obs):
    """Estimator of the probability-weighted moment. From:
    Hosking, J. R. M., and James R. Wallis. 1997.
    “L-Moments.” In Regional Frequency Analysis, 14–43.
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.004.
    """
    return ams.sum(axis=axis) / n_obs


# def b1(ams, rank, axis, n_obs):
#     """Estimator of the probability-weighted moment. From:
#     Hosking, J. R. M., and James R. Wallis. 1997.
#     “L-Moments.” In Regional Frequency Analysis, 14–43.
#     Cambridge: Cambridge University Press.
#     https://doi.org/10.1017/CBO9780511529443.004.
#     """
#     num = rank[1:]
#     denum = n_obs - 1
#     return


# def b2(ams, rank, axis, n_obs):
#     """Estimator of the probability-weighted moment. From:
#     Hosking, J. R. M., and James R. Wallis. 1997.
#     “L-Moments.” In Regional Frequency Analysis, 14–43.
#     Cambridge: Cambridge University Press.
#     https://doi.org/10.1017/CBO9780511529443.004.
#     """
#     return 


def gen_bvalue(ecdf, ams, n_obs, order, axis):
    """Estimation of bvalue not depending on xarray
    """
    pr_sum = ecdf**order * ams.sum(axis=axis)
    return pr_sum / n_obs


def b_value(ds, order):
    """Estimator of the probability-weighted moment
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    n_obs = ds['year'].count()
    axis = ds['annual_max'].get_index('year')
    # Hosking (1990) calls for a specific plotting position
    # Here we use the more common Gringorten, for consistency
    return gen_bvalue(ecdf=ds['ecdf_gringorten'],
                      ams=ds['annual_max'],
                      n_obs=n_obs,
                      order=order,
                      axis=axis)


def sample_L_moments(b0, b1, b2, b3):
    """Sample L-moments
    Hosking, J. R. M., and James R. Wallis. 1997.
    “L-Moments.” In Regional Frequency Analysis, 14–43.
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.004.
    """
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6*b2 - 6*b1 + b0
    l4 = 20*b3 - 30*b2 + 12*b1 - b0
    return l1, l2, l3, l4


def l_ratios(l1, l2, l3, l4):
    """
    """
    LCV = l2 / l1
    Lskewness = l3 / l2
    Lkurtosis = l4 / l2
    return LCV, Lskewness, Lkurtosis


def z_test(shape, n_obs):
    """Test whether the shape parameter is zero.
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    return shape * (n_obs / 0.5633) ** .5


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


def lnt(T):
    return -np.log(1 - 1/T)


def gumbel_quantile(T, loc, scale):
    """Return quantile (i.e, intensity)
    """
    return loc - scale * np.log(lnt(T))


def gev_quantile_nonzero(T, loc, scale, shape):
    """Consider an EV type II if shape<0
    """
    num = scale * (1 - lnt(T)**shape)
    return loc + num / shape


def gev_quantile(T, loc, scale, shape):
    """Overeem, Aart, Adri Buishand, and Iwan Holleman. 2008.
    “Rainfall Depth-Duration-Frequency Curves and Their Uncertainties.”
    Journal of Hydrology 348 (1–2): 124–34.
    https://doi.org/10.1016/j.jhydrol.2007.09.044.
    """
    return xr.where(shape == 0,
                    gumbel_quantile(T, loc, scale),
                    gev_quantile_nonzero(T, loc, scale, shape))
