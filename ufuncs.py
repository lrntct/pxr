# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
# from scipy.special import gamma
import scipy.stats
import numpy as np
import bottleneck
import numba as nb

import helper

fscalar = np.float32
iscalar = np.int32

# Euler-Mascheroni constant
EM = fscalar(0.577215664901532860606512090082)


def ci_gev_func(arr_ams, sampling_idx, n_obs, ci_range, shape_param):
    """Uses njit functions to estimate the confidence interval of GEV parameters using the bootstrap method
    arr_ams is one-dimensional
    """
    # Draw samples. Add dimension n_sample in first position
    arr_samples = arr_ams[sampling_idx]
    ax_year = 1
    # rank samples
    rank = bottleneck.nanrankdata(arr_samples, axis=ax_year).astype(fscalar)
    # fit distribution. ev_apams is a tuple of ndarrays
    ecdf = ecdf_goda(rank, n_obs)
    ev_params = gev_gufunc(arr_samples, ecdf, n_obs,
                                  ax_year, shape=shape_param)
    # Add one axis. Changes shape to (ev_params, samples)
    ev_params = np.array(ev_params)
    # get confidence interval. Changes shape to (quantiles, ev_params)
    c_low = (1 - ci_range) / 2
    c_high = 1 - c_low
    quantiles = np.nanquantile(ev_params, [c_low, c_high], axis=1)
    # print('quantiles', quantiles.shape)
    return quantiles


def perform_analysis(arr_ams, ax_year=0, gev_shape=None):
    """Perform all the analysis on a single cell from Annual Maxima Series (AMS).
    The input is a numpy 2D array (duration x year).
    Do:
    - ranking
    - ecdf calculation
    - estimate parameters of GEV
    - Z test (is shape zero?)
    - compute CDF
    - compute goodness of fit of GEV
    - fit a power law across durations
    """
    # print(arr_ams.shape)
    # print(arr_ams.dtype)
    n_obs = iscalar(arr_ams.shape[ax_year])
    # Rank
    arr_rank = bottleneck.nanrankdata(arr_ams, axis=ax_year).astype(fscalar)
    # print(arr_rank.shape)
    # print(arr_rank.dtype)
    # ECDF
    arr_ecdf = ecdf_goda(arr_rank, n_obs)
    # print(arr_ecdf.shape)
    # print(arr_ecdf.dtype)
    # GEV
    gev_params = gev_gufunc(arr_ams, arr_ecdf, n_obs, ax_year=ax_year, shape=gev_shape)
    # print(gev_params.shape)
    # print(gev_params.dtype)
    arr_loc = gev_params[0]
    arr_scale = gev_params[1]
    arr_shape = gev_params[2]
    # Ztest
    arr_zstat = z_test(arr_shape, n_obs)
    # GEV CDF
    arr_cdf = gev_cdf(arr_ams, *gev_params)
    # Goodness of Fit
    # arr_KS_D = KS_test(arr_ecdf, arr_cdf, ax_year=ax_year)
    # Check results
    arr_tuple = (arr_rank, arr_ecdf, arr_cdf, #arr_KS_D,
                 arr_loc, arr_scale, arr_shape, arr_zstat)
    # for arr in arr_tuple:
    #     print(arr.shape)
    #     print(arr.dtype)
    full_arr = np.stack(arr_tuple)
    # print(full_arr.shape)
    return full_arr


@nb.njit()
def KS_test(ecdf, cdf, ax_year):
    """Retun the Kolmogorov–Smirnov test statistic
    """
    return np.abs(ecdf - cdf).max(axis=ax_year)

# @nb.njit("float32[:, :](float32[:, :], int32)")
@nb.njit()
def ecdf_goda(rank, n_obs):
    return (rank - fscalar(.45)) / n_obs


@nb.njit()
def gumbel_scale(l2):
    return l2 / fscalar(0.6931)  # ln(2)


@nb.njit()
def gumbel_loc(l1, scale):
    return l1 - EM * scale


@nb.vectorize(["float32(float32)", "float64(float64)"])
def gamma(x):
    return math.gamma(x)


@nb.njit()
def gev_scale(l2, shape):
    """Scale parameter of the GEV
    EV type II when shape<0.
    Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    return (l2 * shape) / (gamma(iscalar(1) + shape) * (iscalar(1) - iscalar(2)**-shape))


@nb.njit()
def gev_loc(l1, scale, shape):
    """Location parameter of the GEV
    EV type II when shape<0.
    Hosking, J. R. M., & Wallis, J. R. (1997).
    Appendix: L-moments for some specific distributions.
    In Regional Frequency Analysis (pp. 191–209).
    Cambridge: Cambridge University Press.
    https://doi.org/10.1017/CBO9780511529443.012
    """
    return l1 - scale * ((iscalar(1) - gamma(iscalar(1) + shape)) / shape)


@nb.njit()
def gev_shape(b0, b1, b2):
    """Shape parameter of the GEV
    EV type II when shape<0.
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    c = (iscalar(2)*b1 - b0) / (iscalar(3)*b2 - b0) - fscalar(0.63093)  # ln(2) / ln(3)
    return fscalar(7.859) * c + fscalar(2.9554) * c**iscalar(2)


@nb.jit()
def gev_gufunc(ams, ecdf, n_obs, ax_year, shape=None):
    """
    """
    b0 = np.mean(ams, axis=ax_year)
    # b0 = gen_bvalue(ecdf, ams, n_obs, iscalar(0), ax_year)
    b1 = gen_bvalue(ecdf, ams, n_obs, iscalar(1), ax_year)
    l1 = b0
    l2 = iscalar(2) * b1 - b0
    if shape is not None:
        arr_shape = np.full_like(b0, shape)
    else:
        b2 = gen_bvalue(ecdf, ams, n_obs, iscalar(2), ax_year)
        arr_shape = gev_shape(b0, b1, b2)
    arr_scale = np.where(arr_shape == 0,
                         gumbel_scale(l2),
                         gev_scale(l2, arr_shape))
    arr_loc = np.where(arr_shape == 0,
                       gumbel_loc(l1, arr_scale),
                       gev_loc(l1, arr_scale, arr_shape))
    return arr_loc, arr_scale, arr_shape


@nb.njit()
def gumbel_cdf(x, loc, scale):
    z = (x - loc) / scale
    return fscalar(np.e)**(-fscalar(np.e)**-z)


@nb.njit()
def gev_cdf_nonzero(x, loc, scale, shape):
    """Consider an EV type II if shape<0
    """
    z = (x - loc) / scale
    return fscalar(np.e) ** (-(iscalar(1)-shape*z)**(iscalar(1)/shape))


@nb.njit()
def gev_cdf(x, loc, scale, shape):
    return np.where(shape == 0,
                    gumbel_cdf(x, loc, scale),
                    gev_cdf_nonzero(x, loc, scale, shape))


@nb.njit()
def gen_bvalue(ecdf, ams, n_obs, order, axis):
    """Estimation of bvalue not depending on xarray
    """
    pr_sum = (ecdf**order * ams).sum(axis=axis)
    bvalue = pr_sum / n_obs
    return bvalue


@nb.njit()
def z_test(shape, n_obs):
    """Test whether the shape parameter is zero.
    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    return shape * (iscalar(n_obs) / fscalar(0.5633)) ** fscalar(.5)


@nb.njit()
def lnt(T):
    return -np.log(1 - 1/T)


@nb.njit()
def gumbel_quantile(T, loc, scale):
    """Return quantile (i.e, intensity) for a given return period T in years
    """
    return loc - scale * np.log(lnt(T))


@nb.njit()
def gev_quantile_nonzero(T, loc, scale, shape):
    """Return quantile (i.e, intensity) for a given return period T in years
    Consider an EV type II if shape<0
    """
    num = scale * (1 - lnt(T)**shape)
    return loc + num / shape


@nb.njit()
def gev_quantile(T, loc, scale, shape):
    """Overeem, Aart, Adri Buishand, and Iwan Holleman. 2008.
    “Rainfall Depth-Duration-Frequency Curves and Their Uncertainties.”
    Journal of Hydrology 348 (1–2): 124–34.
    https://doi.org/10.1016/j.jhydrol.2007.09.044.

    T: return period in years
    """
    return np.where(shape == 0,
                    gumbel_quantile(T, loc, scale),
                    gev_quantile_nonzero(T, loc, scale, shape))
