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


# DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/'
DATA_DIR = '../data/MIDAS/'
HOURLY_ERA5 = 'era5_1979-2018_precip.zarr'
HOURLY_MIDAS = 'midas_1979-2018_precip_select.zarr'

AMS_ERA5 = 'era5_1979-2018_ams.zarr'
AMS_MIDAS = 'midas_1979-2018_ams.zarr'
ANNUAL_ERA5_BASENAME = 'era5_1979-2018_ams_{}.zarr'
ANNUAL_MIDAS_BASENAME = 'midas_1979-2018_ams_{}.zarr'


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

LR_RES = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}
# 4 cells: 1 degree
ERA5_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 30*4, 'longitude': 30*4}
# When resolution is 1 degree
ANNUAL_CHUNKS_1DEG = {'year': -1, 'duration': -1, 'latitude': 30, 'longitude': 30}
EXTRACT_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 30, 'longitude': 30}
MIDAS_CHUNKS = {'year': -1, 'duration':-1, 'station': 200}
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
    n_obs = ds_ams['year'].count().item()
    da_ams = ds_ams['annual_max']
    da_ranks = ev_fit.rank_ams(da_ams)
    # Merge arrays in a single dataset, set the dask chunks
    ds = xr.merge([da_ams, da_ranks]).chunk(chunks)
    # Empirical probability
    ds['ecdf_goda'] = ev_fit.ecdf_goda(da_ranks, n_obs)
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
    da_ams = ds['annual_max']
    ds['cdf'] = ev_fit.gev_cdf(da_ams, loc, scale, shape).transpose(*da_ams.dims)
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
    vars_encoding = {v:GEN_FLOAT_ENCODING for v in ds.data_vars.keys()}
    coords_encoding = {k:v for k,v in COORDS_ENCODING.items() if k in ds.coords.keys()}
    encoding = {**vars_encoding, **coords_encoding}
    ds.to_zarr(path, mode='w', encoding=encoding)


def main():
    # Select the source ('ERA5' or 'MIDAS')
    SOURCE = 'MIDAS'

    if SOURCE == 'ERA5':
        temp_res = 1  # temporal resolution in hours
        precip_var = 'precipitation'
        time_var = 'time'
        hourly_path = os.path.join(DATA_DIR, HOURLY_ERA5)
        ams_path = os.path.join(DATA_DIR, AMS_ERA5)
        path_ranked = os.path.join(DATA_DIR, ANNUAL_ERA5_BASENAME.format('ranked'))
        path_gev = os.path.join(DATA_DIR, ANNUAL_ERA5_BASENAME.format('gev'))
        path_gof = os.path.join(DATA_DIR, ANNUAL_ERA5_BASENAME.format('gof'))
        path_scaling = os.path.join(DATA_DIR, ANNUAL_ERA5_BASENAME.format('scaling'))
        chunk_size = ERA5_CHUNKS
    elif SOURCE == 'MIDAS':
        temp_res = 1  # temporal resolution in hours
        precip_var = 'prcp_amt'
        time_var = 'end_time'
        hourly_path = os.path.join(DATA_DIR, HOURLY_MIDAS)
        ams_path = os.path.join(DATA_DIR, AMS_MIDAS)
        path_ranked = os.path.join(DATA_DIR, ANNUAL_MIDAS_BASENAME.format('ranked'))
        path_gev = os.path.join(DATA_DIR, ANNUAL_MIDAS_BASENAME.format('gev'))
        path_gof = os.path.join(DATA_DIR, ANNUAL_MIDAS_BASENAME.format('gof'))
        path_scaling = os.path.join(DATA_DIR, ANNUAL_MIDAS_BASENAME.format('scaling'))
        chunk_size = MIDAS_CHUNKS
    else:
        raise KeyError

    with ProgressBar():
        # Get annual maxima #
        # hourly = xr.open_zarr(hourly_path)
        # ams = step1_annual_maxs_of_roll_mean(hourly, precip_var, time_var, DURATIONS_ALL, temp_res).chunk(chunk_size)
        # print(ams)
        # to_zarr(ams, ams_path)

        # reshape to 1 deg
        # print(ds_trimmed)
        # ds_r = helper.da_pool(ds_trimmed['annual_max'], .25, 1).to_dataset().chunk(ANNUAL_CHUNKS_1DEG)
        # print(ds_r)


        # Rank # For unknown reason, Dask distributed create buggy ECDF.
        # ams = xr.open_zarr(ams_path)
        # print(ams)
        # ds_ranked = step22_rank_ecdf(ams, chunk_size)
        # to_zarr(ds_ranked, path_ranked)

        # Use dask distributed LocalCluster (uses processes instead of threads)
        client = Client()

        # fit EV #
        ds_ranked = xr.open_zarr(path_ranked)#.loc[EXTRACT]
        ds_gev = step23_fit_gev_with_ci(ds_ranked)
        # to_zarr(ds_gev, path_gev)

        # GoF #
        # ds_gev = xr.open_zarr(path_gev)
        ds_gof = step24_goodness_of_fit(ds_gev, chunk_size)
        # to_zarr(ds_gof, path_gof)

        # Scaling #
        # ds_gof = xr.open_zarr(path_gof)#.loc[EXTRACT].chunk(EXTRACT_CHUNKS)
        ds_scaling = step3_scaling_with_ci(ds_gof)
        to_zarr(ds_scaling, path_scaling)


if __name__ == "__main__":
    sys.exit(main())
