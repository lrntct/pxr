# -*- coding: utf8 -*-

import sys
import os
import math

import numpy as np
import bottleneck
import xarray as xr
import numba as nb

import ev_fit


def rd_choice(arr):
    # sample_list = []
    # for i in range(n):
    #     sample = 
    #     sample_list.append(sample)
    # samples = np.concatenate(sample_list, axis=0)
    # print(samples.shape)
    # return tuple(sample_list)
    return np.random.choice(arr, size=len(arr), replace=True)


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

    samples = = xr.apply_ufunc(
        rd_choice,
        ds['annual_max'],
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        # vectorize=True,
        dask='allowed',
        dask='parallelized',
        output_dtypes=[dtype]
        ).rename('annual_max')

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


def ci_ufunc(arr_ams, n_sample, ci_range):
    n_obs = len(arr_ams)
    new_shape = (n_sample, n_obs)
    # Draw samples
    sampling_idx = np.random.randint(n_obs, size=new_shape, dtype='uint16')
    arr_samples = arr_ams[sampling_idx]
    # rank samples
    rank = bottleneck.nanrankdata(arr_samples, axis=1)
    # Gringorten plotting position
    ecdf = (rank - 0.44) / (n_obs + 0.12)
    
    print(ecdf)


def apply_ci(ds, dtype, n_sample=500, ci_range=0.9):
    """
    """
    low, high = xr.apply_ufunc(
        ci_ufunc,
        ds['annual_max'],
        kwargs={'n_sample': n_sample,
                'ci_range': ci_range},
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        vectorize=True,
        dask='allowed',
        # dask='parallelized',
        output_dtypes=[dtype, dtype]
        ).rename('annual_max')
    print(low)


def parallel_rank(ds):
    """Assign a rank to the random sample (necesity for ECDF and therefore LMO fitting)
    """
    ds['rank'] = xr.apply_ufunc(
        bottleneck.nanrankdata,
        ds['annual_max'],
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        # vectorize=True,
        dask='parallelized',
        output_dtypes=[dtype]
        ).rename('rank')
    return ds


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

