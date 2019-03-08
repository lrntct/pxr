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

import ev_fit
import gof
import scaling
import helper


DATA_DIR_ERA = '/home/lunet/gylc4/geodata/ERA5/'
DATA_DIR_MIDAS = '../data/MIDAS/'
HOURLY_ERA5 = 'era5_1979-2018_precip.zarr'
HOURLY_MIDAS = 'midas_1979-2018_precip_select.zarr'

AMS_BASENAME = '{}_1979-2018_ams.zarr'
ANNUAL_BASENAME = '{}_{}-{}_ams_{}.zarr'


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

# Temporal resolution of the input in hours
TEMP_RES = 1
# TEMP_RES = 24

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}
# 4 cells: 1 degree
ERA5_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 20*4, 'longitude': 20*4}
# When resolution is 1 degree
ANNUAL_CHUNKS_1DEG = {'year': -1, 'duration': -1, 'latitude': 30, 'longitude': 30}
EXTRACT_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 30, 'longitude': 30}
MIDAS_CHUNKS = {'year': -1, 'duration':-1, 'station': 1}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
STR_ENCODING = {'dtype': 'U'}
COORDS_ENCODING = {'ci': STR_ENCODING,
                   'ev_param': STR_ENCODING,
                   'scaling_param': STR_ENCODING,
                  }


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
    da_ams = ds_ams['annual_max']
    n_obs = np.isfinite(da_ams).sum(dim='year').rename('n_obs')
    da_ranks = ev_fit.rank_ams(da_ams)
    # Merge arrays in a single dataset, set the dask chunks
    ds = xr.merge([da_ams, da_ranks, n_obs]).chunk(chunks)
    # Empirical probability
    ds['ecdf'] = ev_fit.ecdf(da_ranks, n_obs)
    return ds.chunk(chunks)


def step23_fit_gev_with_ci(ds, n_sample):
    """Estimate GEV parameters and their confidence intervals.
    CI are estimated with the bootstrap method.
    """
    ds['gev'] = ev_fit.fit_gev(ds, DTYPE, n_sample=n_sample, ci_range=[0.9, 0.95, 0.99], shape=-0.114)
    return ds


def step24_goodness_of_fit(ds, chunks):
    """Goodness of fit with the Lilliefors test.
    """
    # Compute the CDF
    loc = ds['gev'].sel(ci='estimate', ev_param='location')
    scale = ds['gev'].sel(ci='estimate', ev_param='scale')
    shape = ds['gev'].sel(ci='estimate', ev_param='shape')
    da_ams = ds['annual_max']
    ds['cdf'] = ev_fit.gev_cdf(da_ams, loc, scale, shape).transpose(*da_ams.dims)
    # Lilliefors
    ds['KS_D'] = gof.KS_test(ds['ecdf'], ds['cdf'])
    ds = gof.lilliefors_Dcrit(ds, chunks)
    return ds


def step3_scaling_with_ci(ds, n_sample):
    """Estimate linear regression and their confidence intervals.
    CI are estimated with the bootstrap method.
    """
    ds['gev_scaling'] = scaling.scaling_gev(ds, DTYPE, n_sample=n_sample, ci_range=[0.9, 0.95, 0.99], shape=-0.114)
    return ds


def to_zarr(ds, path):
    vars_encoding = {v:GEN_FLOAT_ENCODING for v in ds.data_vars.keys()}
    coords_encoding = {k:v for k,v in COORDS_ENCODING.items() if k in ds.coords.keys()}
    encoding = {**vars_encoding, **coords_encoding}
    ds.to_zarr(path, mode='w', encoding=encoding)


def main():
    # Select the source ('ERA5' or 'MIDAS')
    SOURCE = 'era5'
    BS_SAMPLE = 1000
    START = 2000
    END = 2018

    if SOURCE == 'era5':
        temp_res = 1  # temporal resolution in hours
        precip_var = 'precipitation'
        time_var = 'time'
        data_dir = DATA_DIR_ERA
        hourly_path = os.path.join(data_dir, HOURLY_ERA5)
        chunk_size = ERA5_CHUNKS
    elif SOURCE == 'midas':
        temp_res = 1  # temporal resolution in hours
        precip_var = 'prcp_amt'
        time_var = 'end_time'
        data_dir = DATA_DIR_MIDAS
        hourly_path = os.path.join(data_dir, HOURLY_MIDAS)
        chunk_size = MIDAS_CHUNKS
    else:
        raise KeyError('Unknown source: {}'.format(SOURCE))

    ams_path = os.path.join(data_dir, AMS_BASENAME.format(SOURCE))
    path_ranked = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'ranked'))
    path_gev = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'gev'))
    path_gof = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'gof'))
    path_scaling = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'scaling'))

    with ProgressBar():
        # Get annual maxima #
        print('## AMS: {} ##'.format(datetime.now()))
        # ams = step1_annual_maxs_of_roll_mean(hourly, precip_var, time_var, DURATIONS_ALL, temp_res).chunk(chunk_size)
        # print(ams)
        # to_zarr(ams, ams_path)

        # reshape to 1 deg
        # print(ds_trimmed)
        # ds_r = helper.da_pool(ds_trimmed['annual_max'], .25, 1).to_dataset().chunk(ANNUAL_CHUNKS_1DEG)
        # print(ds_r)


        ## Rank # For unknown reason Dask distributed create buggy ECDF.
        # ams = xr.open_zarr(ams_path).sel(year=slice(START, END))
        # print('## Rank: {} ##'.format(datetime.now()))
        # ds_ranked = step22_rank_ecdf(ams, chunk_size)
        # print(ds_ranked)
        # to_zarr(ds_ranked, path_ranked)

        ## For the next steps, use dask distributed LocalCluster (uses processes instead of threads)
        cluster = LocalCluster(n_workers=32, threads_per_worker=1)
        print(cluster)
        client = Client(cluster)

        ## fit EV ##
        print('## Fit EV: {} ##'.format(datetime.now()))
        ds_ranked = xr.open_zarr(path_ranked)#.loc[EXTRACT]
        ds_gev = step23_fit_gev_with_ci(ds_ranked, BS_SAMPLE)
        to_zarr(ds_gev, path_gev)

        ## GoF ##
        print('## Goodness of fit: {} ##'.format(datetime.now()))
        ds_gev = xr.open_zarr(path_gev)
        ds_gof = step24_goodness_of_fit(ds_gev, chunk_size)
        to_zarr(ds_gof, path_gof)

        ## Scaling ##
        print('## Scaling: {} ##'.format(datetime.now()))
        ds_gof = xr.open_zarr(path_gof)#.loc[EXTRACT].chunk(EXTRACT_CHUNKS)
        ds_scaling = step3_scaling_with_ci(ds_gof, BS_SAMPLE)
        print(ds_scaling)
        nan_count = np.isnan(ds_scaling['gev_scaling'].sel(ci='0.005', ev_param='scale', scaling_param='slope')).load()
        print(nan_count)
        to_zarr(ds_scaling, path_scaling)


if __name__ == "__main__":
    sys.exit(main())
