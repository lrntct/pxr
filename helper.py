# -*- coding: utf8 -*-

import math

import xarray as xr
import numpy as np
import numba as nb


@nb.vectorize(["float32(float32)", "float64(float64)"])
def log(x):
    return math.log(x)


@nb.vectorize(["float32(float32)", "float64(float64)"])
def gamma(x):
    return math.gamma(x)


def ci_range_to_qlevels(range_list):
    """get a list of Confidence interval level.
    Return a list of double sided quantile levels
    """
    q_levels = [0.5]  # At least the median
    for r in range_list:
        c_low = (1 - r) / 2
        c_high = 1 - c_low
        q_levels.append(c_low)
        q_levels.append(c_high)
    return sorted(q_levels)

# @nb.njit()
def OLS_jit(x, y, axis=-1):
    """linear regression using the Ordinary Least Squares.
    """
    assert x.shape[axis] == y.shape[axis]
    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)
    slope = (np.sum((x - mean_x) * (y - mean_y), axis=axis, keepdims=True) /
             np.sum((x - mean_x) * (x - mean_x), axis=axis, keepdims=True))
    intercept = mean_y - slope * mean_x
    # coefficient of determination
    fitted = slope * x + intercept
    rsquared = (np.sum(np.square(fitted - mean_y), axis=axis, keepdims=True) /
                np.sum(np.square(y - mean_y), axis=axis, keepdims=True))
    params = np.array([slope, intercept, rsquared])
    return np.squeeze(params, axis=-1)


def get_sampling_idx(n_sample, n_obs):
    sampling_idx = np.random.randint(n_obs, size=(n_sample, n_obs), dtype='uint16')
    # Add the original order as the last sample
    idx_orig = np.arange(n_obs, dtype='uint16')
    idx_orig = np.expand_dims(idx_orig, axis=0)
    return np.concatenate([sampling_idx, idx_orig])


def da_pool(da, old_res, new_res):
    """Pool value from neighbouring cell at resolution old_res
    into a coarser cell at resolution new_res
    """
    assert abs(da['latitude'][1] - da['latitude'][0]) == old_res
    assert abs(da['longitude'][1] - da['longitude'][0]) == old_res
    assert new_res % old_res == 0
    assert len(da['latitude']) % new_res == 0
    assert len(da['longitude']) % new_res == 0

    agg_step = int(new_res / old_res)
    # Transpose the array to ease reshape
    da_t = da.transpose('duration', 'year', 'latitude', 'longitude')
    # Sort coordinates
    da_sorted = da_t.sortby(['latitude', 'longitude'])
    # New coordinates are at center of the aggregation
    start_offset = (new_res - old_res)/2
    lat_start = da_sorted['latitude'][0]
    lat_end = da_sorted['latitude'][-1]
    lon_start = da_sorted['longitude'][0]
    lon_end = da_sorted['longitude'][-1]
    new_lat_coords = np.arange(lat_start + start_offset, lat_end, new_res)
    new_lon_coords = np.arange(lon_start + start_offset, lon_end, new_res)
    # get all the individual points within the new cell
    da_list = []
    for i in range(agg_step):
        for j in range(agg_step):
            da_sel = da_sorted.isel(latitude=slice(i, None, agg_step),
                                    longitude=slice(j, None, agg_step))
            da_sel.coords['latitude'] = new_lat_coords
            da_sel.coords['longitude'] = new_lon_coords
            da_list.append(da_sel)
    # Concatenate along new dimension
    da_neighbours = xr.concat(da_list, dim='neighbours')
    # Stack the dimensions together (create multiindex)
    da_stacked = da_neighbours.stack({'stacked': ['neighbours', 'year']})
    # merge the multiindex into one, rename to year (expected by other functions)
    da_r = da_stacked.reset_index('stacked', drop=True).rename(stacked='year')
    da_r.coords['year'] = range(len(da_r['year']))
    # Reorder the dimensions
    return da_r.transpose('duration', 'year', 'latitude', 'longitude')
