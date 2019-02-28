# -*- coding: utf8 -*-

import math

import xarray as xr
import numpy as np
import numba as nb
import statsmodels.api as sm



@nb.vectorize(["float32(float32)", "float64(float64)"])
def gamma(x):
    return math.gamma(x)


def linregress(func, x, y, dims):
    """x, y: dataArray to use for the regression
    dims: dimension on which to carry out the regression
    """
    # return a tuple of DataArrays
    res = xr.apply_ufunc(func, x, y,
            input_core_dims=[dims, dims],
            output_core_dims=[[] for i in LR_RES],
            vectorize=True,
            dask='allowed',
            output_dtypes=[DTYPE for i in LR_RES]
            )
    return res


def nanlinregress(x, y):
    """wrapper around statsmodels OLS to make it behave like scipy linregress.
    Make use of its capacity to ignore NaN.
    """
    X = sm.add_constant(x)
    try:
        results = sm.OLS(y, X, missing='drop').fit()
    except ValueError:
        slope = np.nan
        intercept = np.nan
        rvalue = np.nan
        pvalue = np.nan
        stderr = np.nan
    else:
        slope = results.params[1]
        intercept = results.params[0]
        rvalue = results.rsquared ** .5
        pvalue = results.pvalues[1]
        stderr = results.bse[1]

    return slope, intercept, rvalue, pvalue, stderr


def OLS(da_x, da_y, dim):
    """Linear regression along the dimension dim.
    Use Ordinary Least Squares.
    """
    mean_x = da_x.mean(dim=dim)
    mean_y = da_y.mean(dim=dim)
    x_diff = da_x - mean_x
    slope = ((x_diff * (da_y - mean_y)).sum(dim=dim) /
             (x_diff**2).sum(dim=dim)).rename('line_slope')
    intercept = (mean_y - slope * mean_x).rename('line_intercept')

    # coefficient of determination
    fitted = slope * da_x + intercept
    rsquared = ((np.square(fitted - mean_y).sum(dim=dim)) /
        (np.square(da_y - mean_y).sum(dim=dim))).rename('rsquared')

    return slope, intercept, rsquared


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
