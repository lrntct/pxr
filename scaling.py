# -*- coding: utf8 -*-

import math

import xarray as xr
import numpy as np
import numba as nb

import ev_fit
import helper


@nb.jit()
def gev_scaling_func(arr_ams, ax_duration, n_sample, log_duration, q_levels, shape_param):
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
        ev_params_list.append(ev_fit.gev_from_samples(arr_years, n_sample, shape_param))
    # Stack the parameters in duration. New shape (gev_params, sample, duration)
    ev_params = np.stack(ev_params_list, axis=-1)
    ax_duration = 2
    # print('ev_params', ev_params.shape)
    # Log transform the parameters
    log_ev_params = helper.log10(ev_params)
    # Add two dimensions to duration to fit the shape of ev_params
    log_duration = log_duration[None, None, :]
    assert len(log_duration.shape) == len(log_ev_params.shape)
    # print('log_duration', log_duration.shape)
    # print('log_ev_params', log_ev_params.shape)
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

