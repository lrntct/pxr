# -*- coding: utf8 -*-

import sys
import os
import math

import numpy as np
import bottleneck
import xarray as xr
import numba as nb

import ev_fit
import ufuncs


def ci_gev(ds, dtype, n_sample=500, ci_range=0.9, shape=None):
    """
    """
    # Random sampling of indices, shared across all cells
    n_obs = len(ds['year'])
    new_shape = (n_sample, n_obs)
    sampling_idx = np.random.randint(n_obs, size=new_shape, dtype='uint16')
    # print(sampling_idx)
    # print(ds['annual_max'])
    da_ci = xr.apply_ufunc(
        ufuncs.ci_gev_func,
        ds['annual_max'],
        kwargs={'sampling_idx': sampling_idx,
                'n_obs': n_obs,
                'ci_range': ci_range,
                'shape_param': shape},
        input_core_dims=[['year']],
        output_core_dims=[['ci', 'ev_param']],
        vectorize=True,
        dask='parallelized',
        # dask='allowed',
        output_dtypes=[dtype],
        output_sizes={'ci': 2, 'ev_param': 3}
        )
    da_ci = da_ci.assign_coords(ci=['low', 'high'],
                                ev_param=['location', 'scale', 'shape'])
    return da_ci.rename('gev_parameters').to_dataset()
