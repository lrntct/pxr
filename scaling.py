# -*- coding: utf8 -*-

import math

import xarray as xr
import numpy as np
import numba as nb

import ev_fit
import helper


@nb.jit()
def gev_scaling_func(arr_ams, ax_duration, n_sample, log_duration, q_levels, shape_param):
    """Estimate the parameters of linear regression on the log transformed GEV parameters and duration.
    Get the confidence interval with the bootstrap technique.
    arr_ams is bi-dimensional
    """
    assert arr_ams.ndim == 2
    # Iterate along duration.
    ev_params_list = []
    for i in range(arr_ams.shape[ax_duration]):
        arr_years = arr_ams[...,i]
        # Find GEV parameters. Resulting shape (gev_params, sample)
        ev_params_list.append(ev_fit.gev_from_samples(arr_years, n_sample, shape_param))
    # Stack the parameters in duration. New shape (gev_params, sample, duration)
    ev_params = np.stack(ev_params_list, axis=-1)
    ax_duration = 2
    # print('ev_params', ev_params.shape)
    # Log transform the parameters
    log_ev_params = helper.log10(ev_params)
    # Add two dimensions to duration to fit the shape of ev_params
    log_duration = np.expand_dims(np.expand_dims(log_duration, axis=0), axis=0)
    # print('log_ev_params', log_ev_params.shape)
    # print('log_duration', log_duration.shape)
    assert len(log_duration.shape) == len(log_ev_params.shape)
    # Fit linear regression. Add the ols params on first axis. Resulting shape (ols, gev, sample)
    ols_params = helper.OLS_jit(log_duration, log_ev_params, axis=ax_duration)
    # print('ols_params', ols_params.shape)
    # Get the parameters from the original sample order (last sample). Add a dim to fit shape of quantiles
    orig_params = np.expand_dims(ols_params[:, :, -1], axis=0)
    # Get confidence interval. Resulting shape (quantiles, ols, gev)
    quantiles = np.nanquantile(ols_params[:,:, :-1], q_levels, axis=-1)
    # print('quantiles', quantiles.shape)
    return np.concatenate([orig_params, quantiles])


def scaling_gev(ds, dtype, n_sample=500, ci_range=[0.95], shape=None):
    """Find the scaling property of the GEV parameters, 
    """
    log_dur = helper.log10(ds['duration'].values)
    # Estimate parameters and CI
    q_levels = helper.ci_range_to_qlevels(ci_range)
    da_ci = xr.apply_ufunc(
        gev_scaling_func,
        ds['annual_max'],
        kwargs={'n_sample': n_sample,
                'ax_duration': 1,
                'log_duration': log_dur,
                'q_levels': q_levels,
                'shape_param': shape},
        input_core_dims=[['year', 'duration']],
        output_core_dims=[['ci', 'scaling_param', 'ev_param']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': len(q_levels)+1, 'scaling_param': 4, 'ev_param': 3}
        )
    q_levels_str = ["{0:.3f}".format(l) for l in q_levels]
    da_ci = da_ci.assign_coords(ci=['estimate'] + q_levels_str,
                                scaling_param=['slope', 'intercept', 'rsquared', 'spearman'],
                                ev_param=['location', 'scale', 'shape'])
    return da_ci.rename('gev_scaling')

