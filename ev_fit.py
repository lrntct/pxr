# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
import numpy as np
import numba as nb

import helper

"""Collection of functions related to extreme values probabilities
"""

DTYPE = 'float32'
fscalar = np.float32
iscalar = np.int32

# Euler-Mascheroni constant
EM = fscalar(0.577215664901532860606512090082)


## Rank and plotting positions ##

def rank_ams(da_ams):
    """Rank the annual maxs in in ascending order
    """
    ranks = xr.apply_ufunc(
        bottleneck.nanrankdata,
        da_ams,
        kwargs={'axis': -1},
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        dask='parallelized',
        output_dtypes=[fscalar]
        ).rename('rank')
    # Maintain order of dimensions
    return ranks.transpose(*da_ams.dims)


@nb.njit()
def ecdf_weibull(rank, n_obs):
    """Return the Weibull plotting position
    Recommended by:
    Makkonen, Lasse. 2006.
    “Plotting Positions in Extreme Value Analysis.”
    Journal of Applied Meteorology and Climatology 45 (2): 334–40.
    https://doi.org/10.1175/JAM2349.1.
    """
    return rank / (n_obs + fscalar(1))


@nb.njit()
def ecdf_gringorten(rank, n_obs):
    """Return the Gringorten plotting position
    """
    return (rank - fscalar(0.44)) / (n_obs + fscalar(.12))


@nb.njit()
def ecdf_landwehr(rank, n_obs):
    """Return the plotting position defined in:
    Hosking, J. R. M. (1990).
    L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics.
    Journal of the Royal Statistical Society: Series B (Methodological), 52(1), 105–124.
    https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """
    return (rank - fscalar(0.35)) / n_obs


@nb.njit()
def ecdf_hosking(rank, n_obs):
    """Return the unbiased estimator suggested by:
    Hosking, J. R. M., and J. R. Wallis. 1995.
    “A Comparison of Unbiased and Plotting-Position Estimators of L Moments.”
    Water Resources Research 31 (8): 2019–25.
    https://doi.org/10.1029/95WR01230.
    """
    return rank / n_obs


@nb.njit()
def ecdf_goda_jit(rank, n_obs):
    """Return the plotting position defined in:
    Goda, Yoshimi. 2011.
    “Plotting-Position Estimator for the L-Moment Method and Quantile Confidence Interval
    for the GEV, GPA, and Weibull Distributions Applied for Extreme Wave Analysis.”
    Coastal Engineering Journal 53 (2): 111–49.
    https://doi.org/10.1142/S057856341100229X.
    """
    return (rank - fscalar(0.45)) / n_obs


def ecdf_goda(rank, n_obs):
    """Return the plotting position defined in:
    Goda, Yoshimi. 2011.
    “Plotting-Position Estimator for the L-Moment Method and Quantile Confidence Interval
    for the GEV, GPA, and Weibull Distributions Applied for Extreme Wave Analysis.”
    Coastal Engineering Journal 53 (2): 111–49.
    https://doi.org/10.1142/S057856341100229X.
    """
    return (rank - 0.45) / n_obs


## GEV fit PWM / L-moments ##


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
def gumbel_scale(l2):
    return l2 / fscalar(0.6931)  # ln(2)


@nb.njit()
def gumbel_loc(l1, scale):
    return l1 - EM * scale


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
    return (l2 * shape) / (helper.gamma(iscalar(1) + shape) * (iscalar(1) - iscalar(2)**-shape))


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
    return l1 - scale * ((iscalar(1) - helper.gamma(iscalar(1) + shape)) / shape)


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


@nb.njit()
def gen_bvalue(ecdf, ams, n_obs, order, axis):
    """Estimation of bvalue not depending on xarray
    """
    pr_sum = (ecdf**order * ams).sum(axis=axis)
    bvalue = pr_sum / n_obs
    return bvalue


@nb.jit()
def gev_pwm(ams, ecdf, n_obs, ax_year, shape=None):
    """Fit the GEV using the Method of Probability-Weighted Moments.
    EV type II when shape<0.

    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
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


def gev_from_samples(arr_ams, ax_year, sampling_idx, n_obs, shape_param):
    """draw sample with replacement and find GEV parameters using the
    Probability-Weighted Moments method.
    """
    # Draw samples. Add dimension n_sample in first position
    arr_samples = arr_ams[sampling_idx]
    ax_year += 1
    # rank samples
    rank = bottleneck.nanrankdata(arr_samples, axis=ax_year).astype(fscalar)
    # fit distribution. ev_apams is a tuple of ndarrays
    ecdf = ecdf_goda_jit(rank, n_obs)
    ev_params = gev_pwm(arr_samples, ecdf, n_obs, ax_year, shape=shape_param)
    # Add one axis. Changes shape to (ev_params, samples)
    return np.array(ev_params)


def gev_func(arr_ams, ax_year, sampling_idx, n_obs, ci_range, shape_param):
    """Estimate the GEV parameters and their confidence interval using the bootstrap method.
    The last row of sample idx is the original order, for the actual parameters estimates.
    arr_ams is one-dimensional
    """
    # Find GEV parameters
    ev_params = gev_from_samples(arr_ams, ax_year, sampling_idx, n_obs, shape_param)
    # Get the parameters from the original sample order (last sample)
    orig_params = np.expand_dims(ev_params[:, -1], axis=0)
    # get confidence interval. Changes shape to (quantiles, ev_params)
    c_low = (1 - ci_range) / 2
    c_high = 1 - c_low
    quantiles = np.nanquantile(ev_params[:, :-1], [c_low, c_high], axis=1)
    # Group parameters from the original sample and their confidence interval
    return np.concatenate([orig_params, quantiles])


def fit_gev(ds, dtype, n_sample=500, ci_range=0.9, shape=None):
    """Fit the GEV
    """
    # Random sampling of indices, shared across all cells
    n_obs = len(ds['year'])
    sampling_idx = helper.get_sampling_idx(n_sample, n_obs)
    # Estimate parameters and CI
    da_ci = xr.apply_ufunc(
        gev_func,
        ds['annual_max'],
        kwargs={'sampling_idx': sampling_idx,
                'ax_year': 0,
                'n_obs': n_obs,
                'ci_range': ci_range,
                'shape_param': shape},
        input_core_dims=[['year']],
        output_core_dims=[['ci', 'ev_param']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': 3, 'ev_param': 3}
        )
    da_ci = da_ci.assign_coords(ci=['value', 'low', 'high'],
                                ev_param=['location', 'scale', 'shape'])
    return da_ci.rename('gev')


## CDF ##

# @nb.njit()
def gumbel_cdf(x, loc, scale):
    z = (x - loc) / scale
    return fscalar(np.e)**(-fscalar(np.e)**-z)


# @nb.njit()
def gev_cdf_nonzero(x, loc, scale, shape):
    """Consider an EV type II if shape<0
    """
    z = (x - loc) / scale
    return fscalar(np.e) ** (-(iscalar(1)-shape*z)**(iscalar(1)/shape))


# @nb.njit()
def gev_cdf(x, loc, scale, shape):
    return xr.where(shape == 0,
                    gumbel_cdf(x, loc, scale),
                    gev_cdf_nonzero(x, loc, scale, shape))


## Functions not used in the analysis ##

# def frechet_cdf(x, loc, scale, shape):
#     """
#     """
#     z = (x - loc) / scale
#     return np.e**(-z)**shape

# def gumbel_mle_wrapper(ams):
#     try:
#         params = scipy.stats.gumbel_r.fit(ams)
#     except RuntimeError:
#         params = (np.nan, np.nan)
#     return params[0], params[1]


# def gumbel_mle_fit(ds, dtype=DTYPE):
#     """Employ scipy stats to find the Gumbel coefficients
#     """
#     loc, scale = xr.apply_ufunc(gumbel_mle_wrapper,
#                                 ds['annual_max'],
#                                 input_core_dims=[['year']],
#                                 output_core_dims=[[], []],
#                                 vectorize=True,
#                                 dask='allowed',
#                                 output_dtypes=[dtype, dtype]
#                                 )
#     return loc.rename('location'), scale.rename('scale')


# def frechet_mom(ds, shape=0.114):
#     """Fit Fréchet (EV type II) using the method of moments and a fixed shape parameter.
#     EV type II when shape>0.
#     Koutsoyiannis, D. (2004).
#     Statistics of extremes and estimation of extreme rainfall: II.
#     Empirical investigation of long rainfall records.
#     Hydrological Sciences Journal, 49(4).
#     https://doi.org/10.1623/hysj.49.4.591.54424
#     """
#     c1 = math.sqrt(helper.gamma(1-2*shape) - helper.gamma(1-shape)**2)
#     c3 = (helper.gamma(1-shape)-1)/shape

#     mean = ds['annual_max'].mean(dim='year')
#     std = ds['annual_max'].std(dim='year')
#     scale = (c1 * std).rename('scale')
#     loc = ((mean/scale) - c3).rename('location')
#     da_shape = xr.full_like(mean, shape).rename('shape')
#     return loc, scale, da_shape


# def sample_L_moments(b0, b1, b2, b3):
#     """Sample L-moments
#     Hosking, J. R. M., and James R. Wallis. 1997.
#     “L-Moments.” In Regional Frequency Analysis, 14–43.
#     Cambridge: Cambridge University Press.
#     https://doi.org/10.1017/CBO9780511529443.004.
#     """
#     l1 = b0
#     l2 = 2 * b1 - b0
#     l3 = 6*b2 - 6*b1 + b0
#     l4 = 20*b3 - 30*b2 + 12*b1 - b0
#     return l1, l2, l3, l4


# def l_ratios(l1, l2, l3, l4):
#     """
#     """
#     LCV = l2 / l1
#     Lskewness = l3 / l2
#     Lkurtosis = l4 / l2
#     return LCV, Lskewness, Lkurtosis

# def gumbel_mom(ds):
#     """Fit Gumbel using the method of moments
#     (Maidment 1993, cited by Bougadis & Adamowki 2006)
#     """
#     magic_number1 = 0.45
#     magic_number2 = 0.7797
#     mean = ds['annual_max'].mean(dim='year')
#     std = ds['annual_max'].std(dim='year')
#     loc = (mean - (magic_number1 * std)).rename('location')
#     scale = (magic_number2 * std).rename('scale')
#     return loc, scale, 0


# def gumbel_iter_linear(ds):
#     """Follow the steps described in:
#     Loaiciga, H. A., & Leipnik, R. B. (1999).
#     Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
#     Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
#     https://doi.org/10.1007/s004770050042
#     """
#     # linearize
#     linearize = lambda a: (np.log(np.log(1/a))).astype(DTYPE)
#     reduced_variable = linearize(ds['ecdf_gringorten'])
#     # First fit. Keep only the two first returning DataArrays
#     estim_slope, estim_intercept = helper.linregress(scipy.stats.linregress,
#                                               ds['annual_max'],
#                                               reduced_variable, ['year'])[:2]
#     # get provisional gumbel parameters
#     loc_prov = -estim_intercept / estim_slope
#     scale_prov = -1 / estim_slope
#     # Analytic probability F(x) from Gumbel CDF
#     analytic_prob = gumbel_cdf(ds['annual_max'], loc_prov, scale_prov)
#     # Get the final location and scale parameters
#     analytic_prob_linear = linearize(analytic_prob)
#     analytic_slope, analytic_intercept = helper.linregress(scipy.stats.linregress,
#                                                     ds['annual_max'],
#                                                     analytic_prob_linear, ['year'])[:2]
#     loc_final = (-analytic_intercept / analytic_slope).rename('location')
#     scale_final = (-1 / analytic_slope).rename('scale')
#     return loc_final, scale_final


# def gev_mle_wrapper(ams):
#     try:
#         params = scipy.stats.genextreme.fit(ams)
#     except RuntimeError:
#         params = (np.nan, np.nan, np.nan)
#     return params[1], params[2], params[0]


# def gev_mle_fit(ds, dtype=DTYPE):
#     """Employ scipy stats to find the GEV coefficients
#     """
#     loc, scale, shape = xr.apply_ufunc(gev_mle_wrapper,
#                                 ds['annual_max'],
#                                 input_core_dims=[['year']],
#                                 output_core_dims=[[], [], []],
#                                 vectorize=True,
#                                 dask='allowed',
#                                 output_dtypes=[dtype, dtype, dtype]
#                                 )
#     return loc.rename('location'), scale.rename('scale'), shape.rename('shape')


# def gumbel_pwm(ds):
#     """
#     Hosking, J. R. M., & Wallis, J. R. (1997).
#     Appendix: L-moments for some specific distributions.
#     In Regional Frequency Analysis (pp. 191–209).
#     Cambridge: Cambridge University Press.
#     https://doi.org/10.1017/CBO9780511529443.012
#     """
#     b0 = b_value(ds, 0)
#     b1 = b_value(ds, 1)
#     l1 = b0
#     l2 = 2 * b1 - b0
#     scale = gumbel_scale(l2)
#     loc = gumbel_loc(l1, scale)
#     # return shape = 0 for consistency
#     shape = xr.full_like(scale, 0).rename('shape')
#     return loc.rename('location'), scale.rename('scale'), shape


# def frechet_pwm(ds):
#     """According to [1], the parameters are estimated the
#     same way as the GEV, the shape is just capped to zero.
#     Fit according to [2].

#     [1] Koutsoyiannis, D. (2004).
#     Statistics of extremes and estimation of extreme rainfall: II.
#     Empirical investigation of long rainfall records.
#     Hydrological Sciences Journal, 49(4).
#     https://doi.org/10.1623/hysj.49.4.591.54424
#     [2] Hosking, J. R. M., & Wallis, J. R. (1997).
#     Appendix: L-moments for some specific distributions.
#     In Regional Frequency Analysis (pp. 191–209).
#     Cambridge: Cambridge University Press.
#     https://doi.org/10.1017/CBO9780511529443.012
#     """
#     b0 = b_value(ds, 0)
#     b1 = b_value(ds, 1)
#     b2 = b_value(ds, 2)
#     l1 = b0
#     l2 = 2 * b1 - b0

#     raw_shape = gev_shape(b0, b1, b2)
#     # Shape must be negative for Fréchet
#     shape = xr.ufuncs.fmin(raw_shape, 0)
#     # shape = xr.where(raw_shape >= 0,
#     #                  xr.full_like(raw_shape, 0),
#     #                  raw_shape)
#     # print(shape)
#     scale = xr.where(shape == 0,
#                      gumbel_scale(l2),
#                      gev_scale(l2, shape))
#     loc = xr.where(shape == 0,
#                    gumbel_loc(l1, scale),
#                    gev_loc(l1, scale, shape))

#     return loc.rename('location'), scale.rename('scale'), shape.rename('shape')