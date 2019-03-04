# -*- coding: utf8 -*-

import math

import xarray as xr
import numpy as np
import numba as nb

import ev_fit
import helper


def gev_scaling_func(arr_ams, ax_year, ax_duration, sampling_idx, n_obs, log_duration, q_levels, shape_param):
    """Estimate the parameters of linear regression on the log transformed GEV parameters and duration.
    Get the confidence interval with the bootstrap technique.
    arr_ams is bi-dimensional
    """
    # Find GEV parameters. Resulting shape (gev, sample, duration)
    ev_params = ev_fit.gev_from_samples(arr_ams, ax_year, sampling_idx, n_obs, shape_param)
    ax_duration += 1
    # Log transform the parameters
    log_ev_params = helper.log(ev_params)
    # Add two dimensions to duration to fit the shape of ev_params
    log_duration = np.expand_dims(np.expand_dims(log_duration, axis=0), axis=0)
    assert len(log_duration.shape) == len(log_ev_params.shape)
    # Fit linear regression. Add the ols params on first axis. Resulting shape (ols, gev, sample, duration)
    ols_params = helper.OLS_jit(log_duration, log_ev_params, axis=ax_duration)
    # print('ols_params', ols_params.shape)
    # Get the parameters from the original sample order (last sample). Add a dim to fit shape of quantiles
    orig_params = np.expand_dims(ols_params[:, :, -1], axis=0)
    # Get confidence interval. Resulting shape (quantiles, ols, gev)
    quantiles = np.nanquantile(ols_params[:,:, :-1], q_levels, axis=-1)
    return np.concatenate([orig_params, quantiles])


def scaling_gev(ds, dtype, n_sample=500, ci_range=[0.95], shape=None):
    """Find the scaling property of the GEV parameters, 
    """
    # Random sampling of indices, shared across all cells
    n_obs = len(ds['year'])
    sampling_idx = helper.get_sampling_idx(n_sample, n_obs)
    log_dur = helper.log(ds['duration'].values)
    # Estimate parameters and CI
    q_levels = helper.ci_range_to_qlevels(ci_range)
    da_ci = xr.apply_ufunc(
        gev_scaling_func,
        ds['annual_max'],
        kwargs={'sampling_idx': sampling_idx,
                'ax_year': 0,
                'ax_duration': 1,
                'n_obs': n_obs,
                'log_duration': log_dur,
                'q_levels': q_levels,
                'shape_param': shape},
        input_core_dims=[['year', 'duration']],
        output_core_dims=[['ci', 'scaling_param', 'ev_param']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': len(q_levels)+1, 'scaling_param': 3, 'ev_param': 3}
        )
    da_ci = da_ci.assign_coords(ci=['estimate'] + q_levels,
                                scaling_param=['slope', 'intercept', 'rsquared'],
                                ev_param=['location', 'scale', 'shape'])
    return da_ci.rename('gev_scaling')

