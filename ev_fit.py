# -*- coding: utf8 -*-

import sys
import os
import math

import xarray as xr
import numpy as np
import numba as nb
import bottleneck

import helper

"""Collection of functions related to extreme values probabilities
"""

DTYPE = 'float32'
fscalar = np.float32
iscalar = np.int32

# Euler-Mascheroni constant
EM = fscalar(0.577215664901532860606512090082)


## Rank, ECDF and plotting positions ##

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


def ecdf(rank, n_obs):
    """Return the ECDF
    Recommended as an unbiased estimator for PWM by:
    Hosking, J. R. M., and J. R. Wallis. 1995.
    “A Comparison of Unbiased and Plotting-Position Estimators of L Moments.”
    Water Resources Research 31 (8): 2019–25.
    https://doi.org/10.1029/95WR01230.
    """
    return rank / n_obs

ecdf_jit = nb.njit(ecdf)


def pp_weibull(rank, n_obs):
    """Return the Weibull plotting position
    Recommended by:
    Makkonen, Lasse. 2006.
    “Plotting Positions in Extreme Value Analysis.”
    Journal of Applied Meteorology and Climatology 45 (2): 334–40.
    https://doi.org/10.1175/JAM2349.1.
    """
    return rank / (n_obs + 1)


def pp_cunnane(rank, n_obs):
    """The Cunnane plotting position.
    Recommended for GEV by:
    Wilks, D. S. (2011).
    Empirical Distributions and Exploratory Data Analysis.
    International Geophysics, 100, 23–70.
    https://doi.org/10.1016/B978-0-12-385022-5.00003-8
    """
    return (rank - 0.4) / (n_obs + 0.2)


@nb.njit()
def pp_gringorten(rank, n_obs):
    """Return the Gringorten plotting position
    """
    return (rank - fscalar(0.44)) / (n_obs + fscalar(.12))


@nb.njit()
def pp_landwehr(rank, n_obs):
    """Return the plotting position defined in:
    Hosking, J. R. M. (1990).
    L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics.
    Journal of the Royal Statistical Society: Series B (Methodological), 52(1), 105–124.
    https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
    """
    return (rank - fscalar(0.35)) / n_obs


@nb.njit()
def pp_goda_jit(rank, n_obs):
    """Return the plotting position defined in:
    Goda, Yoshimi. 2011.
    “Plotting-Position Estimator for the L-Moment Method and Quantile Confidence Interval
    for the GEV, GPA, and Weibull Distributions Applied for Extreme Wave Analysis.”
    Coastal Engineering Journal 53 (2): 111–49.
    https://doi.org/10.1142/S057856341100229X.
    """
    return (rank - fscalar(0.45)) / n_obs


def pp_goda(rank, n_obs):
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


def gev_pwm(ams, ecdf, n_obs, ax_year, shape=None):
    """Fit the GEV using the Method of Probability-Weighted Moments.
    EV type II when shape<0.

    Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985).
    Estimation of the Generalized Extreme-Value Distribution
    by the Method of Probability-Weighted Moments.
    Technometrics, 27(3), 251–261.
    https://doi.org/10.1080/00401706.1985.10488049
    """
    # b0 = np.mean(ams, axis=ax_year)
    b0 = gen_bvalue(ecdf, ams, n_obs, iscalar(0), ax_year)
    b1 = gen_bvalue(ecdf, ams, n_obs, iscalar(1), ax_year)
    l1 = b0
    l2 = iscalar(2) * b1 - b0
    if shape is not None:
        arr_shape = np.atleast_1d(np.full_like(b0, shape))
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


@nb.jit()
def gev_from_samples(arr_ams, n_sample, shape_param):
    """draw sample with replacement and find GEV parameters using the
    Probability-Weighted Moments method.
    """
    assert arr_ams.ndim == 1
    # Remove NaN
    arr_ams = arr_ams[np.isfinite(arr_ams)]
    # Records length is exclusive of NaN
    n_obs = len(arr_ams)
    # print('n_obs', n_obs)
    # Random sampling with replacement of indices
    sampling_idx = helper.get_sampling_idx(n_sample, n_obs)
    # Draw samples. Add dimension n_sample in first position.
    arr_samples = arr_ams[sampling_idx]
    # print(arr_samples.shape)
    ax_year = 1
    # rank samples
    rank = bottleneck.nanrankdata(arr_samples, axis=ax_year).astype(fscalar)
    # fit distribution. ev_params is a tuple of ndarrays.
    ecdf = ecdf_jit(rank, n_obs)
    gev_pwm_njit = nb.njit(gev_pwm)
    ev_params = gev_pwm_njit(arr_samples, ecdf, n_obs, ax_year, shape=shape_param)
    # Add one axis. Changes shape to (ev_params, samples).
    return np.array(ev_params)


@nb.jit()
def gev_fit_scale_func(arr_ams, ax_duration, n_sample, log_duration, q_levels, shape_param):
    """Fit the GEV on n_sample bootstrap samples
    Estimate the parameters of linear regression on the log transformed GEV parameters and duration.
    Estimate GEV parameters from the regression line.
    Get the confidence interval with the bootstrap technique.
    arr_ams is bi-dimensional
    """
    assert arr_ams.ndim == 2
    # Iterate along duration.
    ev_params_list = []
    for i in range(arr_ams.shape[ax_duration]):
        arr_years = arr_ams[...,i]
        # Find GEV parameters. Resulting shape (gev_params, sample)
        ev_params_list.append(gev_from_samples(arr_years, n_sample, shape_param))
    # Stack the parameters in duration. New shape (gev_params, sample, duration)
    ev_params = np.stack(ev_params_list, axis=-1)
    ax_duration = 2
    # print('ev_params', ev_params.shape)
    # Log transform the parameters
    log_ev_params = helper.log10(ev_params)
    # Add two dimensions to duration to fit the shape of ev_params
    log_duration = log_duration[None, None, :]
    assert len(log_duration.shape) == len(log_ev_params.shape)
    # Fit linear regression. Add the ols params on first axis. Resulting shape (ols, gev, sample)
    ols_params = helper.OLS_jit(log_duration, log_ev_params, axis=ax_duration)

    # Estimate GEV parameters from the regression
    slope = ols_params[0, :, :]
    log_intercept = ols_params[1, :, :]
    # Match dims of slope and intercept to duration. New shape (gev_params, sample, duration).
    slope = np.expand_dims(slope, axis=-1)
    log_intercept = np.expand_dims(log_intercept, axis=-1)
    params_from_scaling = 10**(log_intercept + log_duration*slope)
    # GEV shape param is not scaled.
    params_from_scaling[2, :, :] = ev_params[2, :, :]
    # print('params_from_scaling', params_from_scaling.shape)
    assert params_from_scaling.shape == ev_params.shape

    # Match shape and stack EV and OLS. New shape (ols, gev_params, sample, duration, 'source').
    new_shape = list(np.maximum(ols_params[:, :, :, None].shape, ev_params[None, :, :, :].shape))
    new_shape.append(3)  # size of source
    ols_ev = np.full(new_shape, np.nan, dtype=np.float32)
    ols_ev[0, :, :, :, 0] = ev_params
    ols_ev[0, :, :, :, 1] = params_from_scaling
    ols_ev[:, :, :, 0, 2] = ols_params
    # print('ols_ev', ols_ev.shape)

    # Get the parameters from the original sample order (last sample). Add a dim to fit shape of quantiles
    orig_params = np.expand_dims(ols_ev[:, :, -1, :, :], axis=0)
    # print('orig_params', orig_params.shape)
    # Get confidence interval, excluding original sample. Resulting shape (quantiles, ols, gev, duration, 'source')
    ev_quantiles = np.nanquantile(ols_ev[:, :, :-1, :, :], q_levels, axis=2)
    ci = np.concatenate([orig_params, ev_quantiles])
    # print('ci', ci.shape)
    return ci


def gev_fit_scale(ds, dtype, n_sample=500, ci_range=[0.95], shape=None):
    """Estimate GEV parameters.
    Find their scaling parameters.
    Estimate the GEV parameters from the scaling.
    Estimate CI with bootstrap.
    """
    log_dur = helper.log10(ds['duration'].values)
    # Estimate parameters and CI
    q_levels = helper.ci_range_to_qlevels(ci_range)
    da_ci = xr.apply_ufunc(
        gev_fit_scale_func,
        ds['annual_max'],
        kwargs={'n_sample': n_sample,
                'ax_duration': 1,
                'log_duration': log_dur,
                'q_levels': q_levels,
                'shape_param': shape},
        input_core_dims=[['year', 'duration']],
        output_core_dims=[['ci', 'scaling_param',
                           'ev_param', 'duration', 'source']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': len(q_levels)+1,
                      'scaling_param': 4, 'ev_param': 3,
                      'duration': len(log_dur), 'source': 3}
        )
    q_levels_str = ["{0:.3f}".format(l) for l in q_levels]
    da_ci = da_ci.assign_coords(ci=['estimate'] + q_levels_str,
                                scaling_param=['slope', 'intercept', 'rsquared', 'spearman'],
                                ev_param=['location', 'scale', 'shape'],
                                duration=ds['duration'],
                                source=['gev_params', 'gev_scaled', 'gev_scaling'])
    # Detangle the arrays that were joined. Use the first coord to drop the uneeded dimension.
    da_ev_params = da_ci.sel(source='gev_params', scaling_param=da_ci['scaling_param'][0], drop=True).rename('gev')
    da_ev_scaled = da_ci.sel(source='gev_scaled', scaling_param=da_ci['scaling_param'][0], drop=True).rename('gev_scaled')
    da_gev_scaling = da_ci.sel(source='gev_scaling', duration=da_ci['duration'][0], drop=True).rename('gev_scaling')
    return xr.merge([da_ev_params, da_ev_scaled, da_gev_scaling])


## CDF ##

@nb.jit()
def gumbel_cdf(x, loc, scale):
    z = (x - loc) / scale
    return fscalar(np.e)**(-fscalar(np.e)**-z)


@nb.jit()
def gev_cdf_nonzero(x, loc, scale, shape):
    """Consider an EV type II if shape<0
    """
    z = (x - loc) / scale
    return fscalar(np.e) ** (-(1-shape*z)**(1/shape))


def gev_cdf(x, loc, scale, shape):
    return xr.where(shape == 0,
                    gumbel_cdf(x, loc, scale),
                    gev_cdf_nonzero(x, loc, scale, shape))

