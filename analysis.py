# -*- coding: utf8 -*-
import sys
import os
from datetime import datetime, timedelta
import csv
import math

import numpy as np
import numba as nb
import xarray as xr
import dask
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler
from dask.distributed import Client, LocalCluster
import zarr
import scipy.stats

import ev_fit
import gof
import helper


DATA_DIR_ERA = '/home/lunet/gylc4/geodata/ERA5/'
DATA_DIR_ERA_PRECIP = '/home/lunet/gylc4/geodata/ERA5/yearly_zarr'
DATA_DIR_MIDAS = '../data/MIDAS/'
HOURLY_ERA5 = 'era5_1979-2018_precip.zarr'
HOURLY_MIDAS = 'midas_1979-2018_precip_select.zarr'

AMS_BASENAME = '{}_{}-{}_ams.zarr'
ANNUAL_BASENAME = '{}_{}-{}_ams_{}.zarr'


# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32.5, 35))
EXTRACT = dict(latitude=slice(90, 80),
               longitude=slice(0, 10))

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
EXTRACT_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 20, 'longitude': 20}
MIDAS_CHUNKS = {'year': -1, 'duration':-1, 'station': 1}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
STR_ENCODING = {'dtype': 'U'}
COORDS_ENCODING = {'ci': STR_ENCODING,
                   'ev_param': STR_ENCODING,
                   'scaling_param': STR_ENCODING,
                  }


def amax_of_yearly_files(yearly_dir, start, end, precip_var, time_dim, durations, temp_res, chunks):
    """Read yearly records from a given directory.
    For each year, get the AMS.
    """
    years = range(start, end+1)
    precip_dict = {}
    for filename in os.listdir(yearly_dir):
        year_str = filename[:4]
        if int(year_str) in years and 'ams' not in filename:
            file_path = os.path.join(yearly_dir, filename)
            precip = xr.open_zarr(file_path)[precip_var]
            precip_dict[int(year_str)] = precip

    amax_year_list = []
    for year, da_precip in precip_dict.items():
        amax_dur_list = []
        for duration in durations:
            window_size = int(duration / temp_res)
            precip_roll = da_precip.rolling(**{time_dim:window_size}, min_periods=max(int(window_size*.9), 1))
            precip_roll_mean = precip_roll.mean(dim=time_dim, skipna=True)
            amax = precip_roll_mean.max(dim=time_dim, skipna=True)
            amax = amax.expand_dims('duration').rename('annual_max')
            amax.coords['duration'] = [duration]
            amax_dur_list.append(amax)
        amax_year = xr.concat(amax_dur_list, dim='duration').sortby('duration')
        amax_year = amax_year.expand_dims('year').chunk(chunks)
        amax_year.coords['year'] = [year]
        ams_year_path = os.path.join(yearly_dir, '{}_ams.zarr'.format(year))
        print(ams_year_path)
        amax_year.to_dataset().to_zarr(ams_year_path, mode='w')


def concat_amax(yearly_dir, start, end):
    years = range(start, end+1)
    amax_list = []
    for filename in os.listdir(yearly_dir):
        year_str = filename[:4]
        if int(year_str) in years and filename.endswith('ams.zarr'):
            file_path = os.path.join(yearly_dir, filename)
            amax = xr.open_zarr(file_path)['annual_max']
            amax_list.append(amax)

    amax_ds = xr.concat(amax_list, dim='year').sortby('year').to_dataset()
    return amax_ds


def amax_from_file(ds, precip_var, time_dim, durations, temp_res):
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
    return xr.concat(da_list, dim='duration').to_dataset().assign_coords(duration=durations)


def arg_maxs_of_roll_mean(ds, precip_var, time_dim, durations, temp_res):
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


def step12_pole_trim(ds):
    # remove north pole to get a dividable latitude number
    lat_second = ds['latitude'][1].item()
    lat_last = ds['latitude'][-1].item()
    ds = ds.sel(latitude=slice(lat_second, lat_last))
    return(ds)


def step2_rank_ecdf(ds_ams, chunks):
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


def step3_fit_gev_with_ci(ds, n_sample, ev_shape=None):
    """Estimate GEV parameters and their scaling in duration.
    Confidence intervals are estimated with the bootstrap method.
    """
    ds_gev = ev_fit.gev_fit_scale(ds, DTYPE, n_sample=n_sample,
                                  ci_range=[0.9, 0.95, 0.99], shape=ev_shape)
    return xr.merge([ds, ds_gev])


def step4_goodness_of_fit(ds, chunks, ev_shape=None):
    """Goodness of fit.
    """
    # Compute the CDF
    loc = ds['gev'].sel(ci='estimate', ev_param='location')
    scale = ds['gev'].sel(ci='estimate', ev_param='scale')
    shape = ds['gev'].sel(ci='estimate', ev_param='shape')
    da_ams = ds['annual_max']
    ds['cdf'] = ev_fit.gev_cdf(da_ams, loc, scale, shape).transpose(*da_ams.dims)
    # Lilliefors
    ds['KS_D'] = gof.KS_test(ds['ecdf'], ds['cdf'])
    ds = gof.lilliefors_Dcrit(ds, chunks, shape=ev_shape)
    # Filliben
    ds = gof.filliben_test(ds)
    n_obs = int(ds['n_obs'].max())
    filliben_crit = gof.filliben_crit(shape=ev_shape, n_obs=n_obs)
    print(filliben_crit)
    ds['filliben_crit'] = xr.DataArray(filliben_crit,
                                       coords=[[0.05, 0.1]],
                                       dims=['significance_level'])
    return ds


def to_zarr(ds, path):
    vars_encoding = {v:GEN_FLOAT_ENCODING for v in ds.data_vars.keys()}
    coords_encoding = {k:v for k,v in COORDS_ENCODING.items() if k in ds.coords.keys()}
    encoding = {**vars_encoding, **coords_encoding}
    ds.to_zarr(path, mode='w', encoding=encoding)


def main():
    # Select the source ('era5' or 'midas')
    SOURCE = 'era5'
    BS_SAMPLE = 1000
    START = 1979
    END = 2018
    EV_SHAPE = -0.114

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

    ams_path = os.path.join(data_dir, AMS_BASENAME.format(SOURCE, START, END))
    path_ranked = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'ranked'))
    path_gev = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'gev'))
    path_gof = os.path.join(data_dir, ANNUAL_BASENAME.format(SOURCE, START, END, 'gof'))

    with ProgressBar(), np.warnings.catch_warnings():
        # Those warnings are expected.
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        np.warnings.filterwarnings('ignore', r'invalid value encountered in log10')

        # Get annual maxima #
        # print('## AMS: {} ##'.format(datetime.now()))
        # amax_of_yearly_files(DATA_DIR_ERA_PRECIP, START, END, precip_var, time_var, DURATIONS_ALL, temp_res, chunk_size)
        # ams = concat_amax(DATA_DIR_ERA_PRECIP, START, END).chunk(chunk_size)
        # ams = step1_annual_maxs_of_roll_mean(hourly, precip_var, time_var, DURATIONS_ALL, temp_res).chunk(chunk_size)
        # print(ams)
        # to_zarr(ams, ams_path)

        # reshape to 1 deg
        # ds_r = helper.da_pool(ds_trimmed['annual_max'], .25, 1).to_dataset().chunk(ANNUAL_CHUNKS_1DEG)
        # print(ds_r)

        ## Rank # For unknown reason Dask distributed create buggy ECDF.
        # ams = xr.open_zarr(ams_path).sel(year=slice(START, END))
        # print('## Rank: {} ##'.format(datetime.now()))
        # ds_ranked = step2_rank_ecdf(ams, chunk_size)
        # print(ds_ranked)
        # to_zarr(ds_ranked, path_ranked)

        ## For the next steps, use dask distributed LocalCluster (uses processes instead of threads)
        cluster = LocalCluster(n_workers=32, threads_per_worker=1)
        print(cluster)
        client = Client(cluster)

        ## fit EV ##
        # print('## Fit EV: {} ##'.format(datetime.now()))
        # ds_ranked = xr.open_zarr(path_ranked)#.loc[EXTRACT].chunk(EXTRACT_CHUNKS)
        # ds_gev = step3_fit_gev_with_ci(ds_ranked, BS_SAMPLE, ev_shape=EV_SHAPE)
        # print(ds_gev)
        # to_zarr(ds_gev, path_gev)

        ## GoF ##
        print('## Goodness of fit: {} ##'.format(datetime.now()))
        ds_gev = xr.open_zarr(path_gev)
        ds_gof = step4_goodness_of_fit(ds_gev, chunk_size, ev_shape=EV_SHAPE)
        print(ds_gof)
        to_zarr(ds_gof, path_gof)


if __name__ == "__main__":
    sys.exit(main())
