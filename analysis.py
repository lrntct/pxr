# -*- coding: utf8 -*-
import sys
import os
import copy
from datetime import datetime, timedelta
import csv

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr
import scipy.stats

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_2000-2012_precip_big_chunks.zarr'
ANNUAL_FILE = 'era5_2000-2012_precip_annual_max.zarr'
ANNUAL_FILE_RANK = 'era5_2000-2012_precip_ranked_test.zarr'
ANNUAL_FILE_GUMBEL = 'era5_2000-2012_precip_gumbel_test.nc'
ANNUAL_FILE_GRADIENT = 'era5_2000-2012_precip_gradient_test.nc'

LOG_FILENAME = 'Analysis_log_{}.csv'.format(str(datetime.now()))

# Coordinates of study sites
KAMPALA = (0.317, 32.616)
KISUMU = (0.1, 34.75)
# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32.5, 35))
EXTRACT = dict(latitude=slice(0, -2),
               longitude=slice(0, 2))
# Spatial resolution in degree - used for match coordinates
SPATIAL_RES = 0.25

# Event durations in hours - has to be adjusted to temporal resolution for the moving window
DURATIONS = [i+1 for i in range(24)] + [i for i in range(24+6,48+6,6)] + [i*24 for i in [5,10,15]]
TEMP_RES = 1  # Temporal resolution in hours

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 8, 'longitude': 8}
ANNUAL_CHUNKS = {'year': -1, 'duration':1, 'latitude': 45*4, 'longitude': 45*4}  # 4 cells: 1 degree
ANNUAL_ENCODING = {'annual_max': {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                   'duration': {'dtype': DTYPE},
                   'latitude': {'dtype': DTYPE},
                   'longitude': {'dtype': DTYPE}}


def round_partial(value):
    return round(value / SPATIAL_RES) * SPATIAL_RES


def step1_annual_maxs_of_roll_mean(ds, durations, temp_res):
    """for each rolling winfows size:
    compute the annual maximum of a moving mean
    return a dataset with the durations as variables
    """
    annual_maxs = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip = ds.precipitation
        precip_roll_mean = precip.rolling(time=window_size).mean(dim='time')
        annual_max = precip_roll_mean.groupby('time.year').max(dim='time')
        annual_max.name = 'annual_max'
        da = annual_max.expand_dims('duration')
        da.coords['duration'] = [duration]
        annual_maxs.append(da)
    return xr.concat(annual_maxs, 'duration').chunk(ANNUAL_CHUNKS)


def linregress_ufunc(x, y, param):
    return scipy.stats.linregress(x, y)._asdict()[param]


def linregress(ds, x, y, prefix, dims):
    """ds: xarray dataset
    x, y: name of variables to use for the regression
    prefix: to be added before the indivudual result names
    """
    lr_params = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    array_names = ['{}_{}'.format(prefix, n) for n in lr_params]
    arr_res = []
    for param_name in lr_params:
        arr_res.append(xr.apply_ufunc(linregress_ufunc,
                                      ds[x], ds[y], kwargs={'param':param_name},
                                      input_core_dims=[dims, dims],
                                      output_core_dims=[[]],
                                      vectorize=True,
                                      dask='parallelized',
                                      output_dtypes=[DTYPE]
                                      ))
    # add the data to the existing dataset
    for arr_name, arr in zip(array_names, arr_res):
        ds[arr_name] = arr


def double_log(arr):
    return (np.log(np.log(1/arr))).astype(DTYPE)


def step21_ranking(annual_maxs):
    """Rank the annual maxs in time, in descending order
    return a Dataset
    """
    n_obs = annual_maxs.count(dim='year')
    # rank is working in ascending order
    ranks = (n_obs - annual_maxs.load().rank(dim='year')).rename('rank').astype(DTYPE)
    return xr.merge([annual_maxs, ranks])


def step22_gumbel_fit(ds):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    n_obs = ds.annual_max.count(dim='year')
    # Estimate probability F{x} with plotting positions
    ds['plot_pos'] = (ds['rank'] / (n_obs+1)).astype(DTYPE)
    ds['gumbel_prov'] = double_log(ds['plot_pos'])
    # First fit
    linregress(ds, 'annual_max', 'gumbel_prov', 'prov_lr', ['year'])
    # get provisional gumbel parameters
    ds['loc_prov'] = -ds['prov_lr_intercept']/ds['prov_lr_slope']
    ds['scale_prov'] = -1/ds['prov_lr_slope']
    # Analytic probability F(x) from Gumbel CDF
    z = (ds['annual_max'] - ds['loc_prov']) / ds['scale_prov']
    ds['gumbel_cdf'] = np.e**(-np.e**-z)
    # Get the final location and scale parameters
    ds['gumbel_final'] = double_log(ds['gumbel_cdf'])
    linregress(ds, 'annual_max', 'gumbel_final', 'final_lr', ['year'])
    ds['loc_final'] = -ds['final_lr_intercept']/ds['final_lr_slope']
    ds['scale_final'] = -1/ds['final_lr_slope']
    return ds


def step3_duration_gradient(ds):
    """Take a Dataset as input containing the fitted gumbel parameters
    Fit a linear regression on the log of the parameters and the log of the duration
    Keep the regression parameters as variables
    """
    ds['log_duration'] = xr.ufuncs.log10(ds['duration'])
    ds['log_location'] = xr.ufuncs.log10(ds['loc_final'])
    ds['log_scale'] = xr.ufuncs.log10(ds['scale_final'])
    linregress(ds, 'log_duration', 'log_location', 'loc_lr', ['duration'])
    linregress(ds, 'log_duration', 'log_scale', 'scale_lr', ['duration'])


def benchmark(ds):
    """Run the gumbel fit for a number of extract sizes
    print result to stdout
    """
    duration_list = []
    sizes = [(i+1)*5 for i in range(5)]
    sizes_sq = [i*i for i in sizes]
    for degrees in sizes:
        locator = dict(latitude=slice(degrees, 0),  # Latitudes are in descending order
                       longitude=slice(0, degrees))
        sel = ds.loc[locator]
        start = datetime.now()
        step2_gumbel_fit(sel)
        duration = datetime.now() - start
        duration_list.append(duration)
    dur_sec = [d.total_seconds() for d in duration_list]
    print({k:v for k, v in zip(sizes_sq, dur_sec)})


def logger(fields):
    log_file_path = os.path.join(DATA_DIR, LOG_FILENAME)
    with open(log_file_path, mode='a') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(fields)


def main():
    kampala_locator = {'latitude': round_partial(KAMPALA[0]),
                       'longitude': round_partial(KAMPALA[1]),
                       #'duration': 6
                       }
    # Log file
    logger(['operation', 'timestamp', 'cumul_sec'])

    with ProgressBar():
        start_time = datetime.now()
        # Load hourly data #
        # logger(['start computing annual maxima', str(start_time), 0])
        # hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
        # hourly = xr.open_zarr(hourly_path).chunk(HOURLY_CHUNKS)
        # hourly_extract = hourly.loc[EXTRACT]
        # print(hourly)

        # Get annual maxima #
        # annual_maxs = step1_annual_maxs_of_roll_mean(hourly, DURATIONS, TEMP_RES)
        amax_path = os.path.join(DATA_DIR, ANNUAL_FILE)
        # annual_maxs.to_dataset().to_zarr(amax_path, mode='w', encoding=ANNUAL_ENCODING)

        annual_maxs = xr.open_zarr(amax_path)['annual_max'].loc[EXTRACT]

        # Do the ranking
        logger(['start ranking annual maxima', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        ds_ranked = step21_ranking(annual_maxs)
        logger(['start writing ranks', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        rank_encoding = copy.deepcopy(ANNUAL_ENCODING)
        rank_encoding['rank'] = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
        rank_path = os.path.join(DATA_DIR, ANNUAL_FILE_RANK)
        ds_ranked.to_zarr(rank_path, mode='w', encoding=rank_encoding)
        print(ds_ranked)

        # fit Gumbel #
        logger(['start gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        ds_fitted = step22_gumbel_fit(ds_ranked)
        logger(['start writting results of gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        gumbel_path = os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL)
        ds_fitted.chunk(ANNUAL_CHUNKS).to_netcdf(gumbel_path, mode='w')

        # fit duration scaling #
        # logger(['start duration scaling fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # step3_duration_gradient(ds_fitted)
        # logger(['start writing duration scaling', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # gradient_path = os.path.join(DATA_DIR, ANNUAL_FILE_GRADIENT)
        # ds_fitted.chunk(ANNUAL_CHUNKS).to_netcdf(gradient_path, mode='w')
        logger(['complete', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        print(ds_fitted)


if __name__ == "__main__":
    sys.exit(main())
