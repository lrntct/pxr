# -*- coding: utf8 -*-
import sys
import os
from datetime import datetime, timedelta
import csv
import math

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.distributed import Client, LocalCluster
import zarr
import scipy.stats

from bokeh.io import export_png

import ev_fit
import gof
import scaling
import helper


DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/'
# DATA_DIR = '../data/MIDAS/'
# HOURLY_FILE1 = 'midas_2000-2017_precip_pairs.nc'
HOURLY_FILE = 'era5_1979-2018_precip.zarr'

AMS_FILE = 'era5_1979-2018_ams.zarr'
ANNUAL_FILE_BASENAME = 'era5_1979-2018_ams_{}.zarr'
# ANNUAL_FILE_GRADIENT = 'era5_2000-2017_precip_gradient.zarr'
# ANNUAL_FILE_SCALING = 'midas_2000-2017_precip_pairs_scaling.nc'

LOG_FILENAME = 'Analysis_log_{}.csv'.format(str(datetime.now()))

# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32.5, 35))
EXTRACT = dict(latitude=slice(90, 45),
               longitude=slice(0, 45))

# Event durations in hours - has to be adjusted to temporal resolution for the moving window
# Selected to be equally spaced on a log scale. Manually adjusted from a call to np.geomspace()
DURATIONS_SUBDAILY = [1, 2, 3, 4, 6, 8, 10, 12, 18, 24]
DURATIONS_DAILY = [24, 48, 72, 96, 120, 144, 192, 240, 288, 360]
# use fromkeys to remove duplicate. need py >= 3.6 to preserve order
DURATIONS_ALL = list(dict.fromkeys(DURATIONS_SUBDAILY + DURATIONS_DAILY))

DURATION_DICT = {'all': DURATIONS_ALL, 'daily': DURATIONS_DAILY, 'subdaily': DURATIONS_SUBDAILY}
# DURATION_DICT = {'daily': DURATIONS_DAILY}
# Temporal resolution of the input in hours
TEMP_RES = 1
# TEMP_RES = 24

LR_RES = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}
# 4 cells: 1 degree
ANNUAL_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 30*4, 'longitude': 30*4}
# When resolution is 1 degree
ANNUAL_CHUNKS_1DEG = {'year': -1, 'duration': -1, 'latitude': 30, 'longitude': 30}
EXTRACT_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 30, 'longitude': 30}
GAUGES_CHUNKS = {'year': -1, 'duration':-1, 'station': 200}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
ANNUAL_ENCODING = {'annual_max': GEN_FLOAT_ENCODING,
                   'duration': {'dtype': DTYPE},
                   'latitude': {'dtype': DTYPE},
                   'longitude': {'dtype': DTYPE}}


def step1_annual_maxs_of_roll_mean(ds, precip_var, time_dim, durations, temp_res):
    """for each rolling windows size:
    compute the annual maximum of a moving mean
    return an array with the durations as a new dimension
    """
    da_list = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip = ds[precip_var]
        precip_roll_mean = precip.rolling(**{time_dim:window_size}, min_periods=max(int(window_size*.9), 1)).mean(dim=time_dim, skipna=True)
        annual_max = precip_roll_mean.groupby('{}.year'.format(time_dim)).max(dim=time_dim, skipna=True).rename('annual_max')
        da_list.append(annual_max)
    return xr.concat(da_list, 'duration').to_dataset().assign_coords(duration=durations)


def step11_arg_maxs_of_roll_mean(ds, precip_var, time_dim, durations, temp_res):
    """for each rolling windows size:
    compute the location of the annual maximum of a moving mean
    """
    da_list = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip = ds[precip_var]
        precip_roll_mean = precip.rolling(**{time_dim:window_size}, min_periods=max(int(window_size*.9), 1)).mean(dim=time_dim, skipna=True)
        arg_max = precip_roll_mean.groupby('{}.year'.format(time_dim)).argmax(dim=time_dim, skipna=True).rename('argmax')
        da_list.append(arg_max)
    return xr.concat(da_list, 'duration')


def step21_pole_trim(ds):
    # remove north pole to get a dividable latitude number
    lat_second = ds['latitude'][1].item()
    lat_last = ds['latitude'][-1].item()
    ds = ds.sel(latitude=slice(lat_second, lat_last))
    return(ds)


def step22_rank_ecdf(ds_ams, chunks):
    """Compute the rank of AMS and the empirical probability
    """
    n_obs = ds_ams['year'].count().item()
    da_ams = ds_ams['annual_max']
    da_ranks = ev_fit.rank_ams(da_ams)
    # Merge arrays in a single dataset, set the dask chunks
    ds = xr.merge([da_ams, da_ranks]).chunk(chunks)
    # Empirical probability
    ds['ecdf_goda'] = xr.apply_ufunc(
        ev_fit.ecdf_goda,
        da_ranks,
        kwargs={'n_obs': n_obs},
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[DTYPE]
        ).transpose(*da_ams.dims)
    return ds


def step23_fit_gev_with_ci(ds):
    """Estimate GEV parameters and their confidence intervals.
    CI are estimated with the bootstrap method.
    """
    ds['gev'] = ev_fit.fit_gev(ds, DTYPE, n_sample=1000, shape=-0.114)
    return ds


def step24_goodness_of_fit(ds, chunks):
    """Goodness of fit with the Lilliefors test.
    """
    # Compute the CDF
    loc = ds['gev'].sel(ci='value', ev_param='location')
    scale = ds['gev'].sel(ci='value', ev_param='scale')
    shape = ds['gev'].sel(ci='value', ev_param='shape')
    ds['cdf'] = ev_fit.gev_cdf(ds['annual_max'], loc, scale, shape)
    # Lilliefors
    ds['KS_D'] = gof.KS_test(ds['ecdf_goda'], ds['cdf'])
    ds = gof.lilliefors_Dcrit(ds, chunks)
    return ds


def step3_scaling_with_ci(ds):
    """Estimate linear regression and their confidence intervals.
    CI are estimated with the bootstrap method.
    """
    ds['gev_scaling'] = scaling.scaling_gev(ds, DTYPE, n_sample=1000, shape=-0.114)
    return ds


def to_zarr(ds, path):
    encoding = {v:GEN_FLOAT_ENCODING for v in ds.data_vars.keys()}
    ds.to_zarr(path, mode='w', encoding=encoding)


def main():
    with ProgressBar():
        # Load hourly data #
        # hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
        # hourly = xr.open_zarr(hourly_path)

        # Get annual maxima #
        # annual_maxs = step1_annual_maxs_of_roll_mean(hourly1, 'prcp_amt', 'end_time', DURATIONS_ALL, TEMP_RES)#.chunk(ANNUAL_CHUNKS)
        # ams = step1_annual_maxs_of_roll_mean(hourly, 'precipitation', 'time', DURATIONS_ALL, TEMP_RES).chunk(ANNUAL_CHUNKS)
        # amax_path = os.path.join(DATA_DIR, AMS_FILE)
        # encoding = {v:GEN_FLOAT_ENCODING for v in ams.data_vars.keys()}
        # ams.to_zarr(amax_path, mode='w', encoding=encoding)

        # argmax = step11_arg_maxs_of_roll_mean(hourly, 'precipitation', 'time', DURATIONS_ALL, TEMP_RES).chunk(ANNUAL_CHUNKS).to_dataset()
        # argmax_path = os.path.join(DATA_DIR, ANNUAL_FILE_BASENAME.format('argmax'))
        # encoding = {v:GEN_FLOAT_ENCODING for v in argmax.data_vars.keys()}
        # print(argmax.load())
        # argmax.to_zarr(argmax_path, mode='w', encoding=encoding)

        # reshape to 1 deg
        # print(ds_trimmed)
        # ds_r = helper.da_pool(ds_trimmed['annual_max'], .25, 1).to_dataset().chunk(ANNUAL_CHUNKS_1DEG)
        # print(ds_r)

        ams_path = os.path.join(DATA_DIR, AMS_FILE)
        path_ranked = os.path.join(DATA_DIR, ANNUAL_FILE_BASENAME.format('ranked'))
        path_gev = os.path.join(DATA_DIR, ANNUAL_FILE_BASENAME.format('gev'))
        path_gof = os.path.join(DATA_DIR, ANNUAL_FILE_BASENAME.format('gof'))
        path_scaling = os.path.join(DATA_DIR, ANNUAL_FILE_BASENAME.format('scaling'))

        ams = xr.open_zarr(ams_path)

        # Rank # 
        # ds_ranked = step22_rank_ecdf(ams, ANNUAL_CHUNKS)
        # print(ds_ranked)
        # to_zarr(ds_ranked, path_ranked)

        # fit EV #
        ds_ranked = xr.open_zarr(path_ranked)#.loc[EXTRACT]
        ds_gev = step23_fit_gev_with_ci(ds_ranked)
        to_zarr(ds_gev, path_gev)

        # GoF #
        ds_gev = xr.open_zarr(path_gev)
        ds_gof = step24_goodness_of_fit(ds_gev, ANNUAL_CHUNKS)
        to_zarr(ds_gof, path_gof)

        # Scaling #
        ds_gof = xr.open_zarr(path_gof)#.loc[EXTRACT].chunk(EXTRACT_CHUNKS)
        ds_scaling = step3_scaling_with_ci(ds_gof)
        to_zarr(ds_scaling, path_scaling)



if __name__ == "__main__":
    # Use dask distributed LocalCluster (uses processes instead of threads)
    cluster = LocalCluster()
    print(cluster)
    client = Client(cluster)
    sys.exit(main())
