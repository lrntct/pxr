# -*- coding: utf8 -*-

import sys
import os
import math

import numpy as np
import bottleneck
import xarray as xr
import numba as nb

import ev_fit


# @nb.guvectorize(["void(float32[:], float32[], float32[:, :])"], '(n), () ->(m, n)', target="parallel", nopython=True)
# def ji_rd_choice(arr, n_sample, arr_res):
#     for i in n_sample:
#         np.random.choice(arr, size=len(arr), replace=True)
#     return


@nb.njit(parallel=True)
def rd_choice(arr, n_sample):
    return np.random.choice(arr, size=(n_sample, len(arr)), replace=True)


def draw_samples(ds, dtype, n_sample=500):
    """Draw random samples with replacement from the AMS
    """
    # orig_shape = ds['annual_max'].shape
    # new_shape = tuple([n_sample]+[i for i in orig_shape])
    # print(orig_shape)
    # print(new_shape)
    # n_obs = len(ds['year'])
    # sampling_idx = np.random.randint(n_obs, size=new_shape, dtype='uint16')
    # print(sampling_idx.shape)
    # arr_samples = ds['annual_max'].data[sampling_idx]
    # print(arr_samples.shape)

    # da_ams = ds['annual_max']
    samples = xr.apply_ufunc(
        rd_choice,
        ds['annual_max'],
        kwargs={'n_sample': n_sample},
        input_core_dims=[['year']],
        output_core_dims=[['bootstrap_sample', 'year']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[dtype],
        output_sizes={'bootstrap_sample':n_sample}
        ).rename('annual_max')
    print(samples)
    return samples.to_dataset()

    # da_samples = 
    # samples = data[idx]
    # sample_list = []
    # for i in range(n):
    #     sample = xr.apply_ufunc(
    #         rd_choice,
    #         ds['annual_max'],
    #         input_core_dims=[['year']],
    #         output_core_dims=[['year']],
    #         vectorize=True,
    #         # dask='allowed',
    #         dask='parallelized',
    #         output_dtypes=[dtype]
    #         ).rename('annual_max')
    #     sample = sample.expand_dims('bootstrap_iter')
    #     sample.coords['bootstrap_iter'] = [i]
    #     sample_list.append(sample)
    # da_bootstrap = xr.concat(sample_list, dim='bootstrap_iter')
    # print(da_bootstrap)
    # return da_bootstrap.to_dataset()


# @nb.njit()
def ci_func(arr_ams, sampling_idx, n_obs, ci_range, shape_param):
    # shape of sampling_idx is (n_sample, n_obs)
    ax_year = 1 
    # Draw samples
    arr_samples = arr_ams[sampling_idx]
    # rank samples
    # rank = arr_samples.argsort(axis=ax_year).argsort(axis=ax_year)
    rank = bottleneck.nanrankdata(arr_samples, axis=ax_year)
    # fit distribution. ev_apams is a tuple of ndarrays
    ecdf = ev_fit.ecdf_gringorten(rank, n_obs)
    ev_params = ev_fit.gev_gufunc(arr_samples, ecdf, n_obs,
                                  ax_year, shape=shape_param)
    # Add one axis. Changes shape to (ev_params, samples, n_obs)
    ev_params = np.array(ev_params)
    # get confidence interval. Changes shape to (quantiles, ev_params, n_obs)
    c_low = (1 - ci_range) / 2
    c_high = 1 - c_low
    quantiles = np.nanquantile(ev_params, [c_low, c_high], axis=1)
    return quantiles


def ci_gev(ds, dtype, n_sample=500, ci_range=0.9, shape=None):
    """
    """
    # Random sampling of indices, shared across all cells
    n_obs = len(ds['year'])
    new_shape = (n_sample, n_obs)
    sampling_idx = np.random.randint(n_obs, size=new_shape, dtype='uint16')
    # print(sampling_idx)
    # print(ds['annual_max'])
    ci_range = xr.apply_ufunc(
        ci_func,
        ds['annual_max'],
        kwargs={'sampling_idx': sampling_idx,
                'n_obs': n_obs,
                'ci_range': ci_range,
                'shape_param': shape},
        input_core_dims=[['year']],
        output_core_dims=[['ci', 'ev_param', 'year']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': 2, 'ev_param': 3}
        )
    # print(low)
    return ci_range.to_dataset()


def fit_distrib(ds, func, **kwargs):
    n_obs = len(ds['year'])
    # Empirical CDF
    ds['ecdf_gringorten'] = ev_fit.ecdf_gringorten(ds['rank'], n_obs)
    # Fit distribution
    param_list = []
    for da_param in func(ds, **kwargs):
        da_param = da_param.expand_dims('ev_param')
        da_param.coords['ev_param'] = [da_param.name]
        param_list.append(da_param.rename('ev_parameter'))
    ds['ev_parameter'] = xr.concat(param_list, dim='ev_param')
    return ds


def ci(ds, ci_range=0.9):
    """Estimate the confidence interval (CI) using a bootstrap method, whereby:
    - Keep the quantiles of each results from func as CI
    """
    low = (1-ci_range) / 2
    high = 1 - low
    q_values = ds['ev_parameter'].quantile([low, high], dim='bootstrap_iter')
    print(q_values)
    return q_values

