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

import config
import ev_fit
import gof
import helper

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
MIDAS_CHUNKS = {'year': -1, 'duration':-1, 'station': -1}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
STR_ENCODING = {'dtype': 'U'}
COORDS_ENCODING = {'ci': STR_ENCODING,
                   'ev_param': STR_ENCODING,
                   'scaling_param': STR_ENCODING,
                   'src_name': STR_ENCODING,
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
    source = config.analysis['source']
    BS_SAMPLE = config.analysis['bootstrap_samples']
    start = config.analysis['start']
    end = config.analysis['end']
    EV_SHAPE = config.analysis['ev_shape']

    data_dir = config.data_dir[source]
    hourly_path = os.path.join(data_dir, config.hourly_filename[source])

    with ProgressBar(), np.warnings.catch_warnings():
        # Those warnings are expected.
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        np.warnings.filterwarnings('ignore', r'invalid value encountered in log10')

        # Get annual maxima #
        # print('## AMS: {} ##'.format(datetime.now()))
        # if source == 'era5':
        #     chunk_size = ERA5_CHUNKS
        #     # Due to its size, era5 data is split in files by year
        #     amax_of_yearly_files(hourly_path, start, end,
        #                          precip_var='precipitation', time_dim='time',
        #                          durations=DURATIONS_ALL, temp_res=1, chunks=chunk_size)
        #     ams = concat_amax(hourly_path, start, end).chunk(chunk_size)
        # elif source == 'midas':
        #     chunk_size = MIDAS_CHUNKS
        #     hourly = xr.open_zarr(hourly_path)
        #     ams = amax_from_file(hourly, precip_var='prcp_amt', time_dim='end_time',
        #                          durations=DURATIONS_ALL, temp_res=1).chunk(chunk_size)
        # else:
        #     raise KeyError('Unknown source: {}'.format(source))
        # print(ams)
        # print(config.path_ams)
        # to_zarr(ams, config.path_ams)

        ## Rank # For unknown reason Dask distributed create buggy ECDF.
        # ams = xr.open_zarr(config.path_ams).sel(year=slice(start, end))
        # print('## Rank: {} ##'.format(datetime.now()))
        # ds_ranked = step2_rank_ecdf(ams, chunk_size)
        # print(ds_ranked)
        # to_zarr(ds_ranked, config.path_ranked)

        ## For the next steps, use dask distributed LocalCluster (uses processes instead of threads)
        cluster = LocalCluster(n_workers=32, threads_per_worker=1)
        print(cluster)
        client = Client(cluster)

        ## fit EV ##
        print('## Fit EV: {} ##'.format(datetime.now()))
        ds_ranked = xr.open_zarr(config.path_ranked)#.loc[EXTRACT].chunk(EXTRACT_CHUNKS)
        ds_gev = step3_fit_gev_with_ci(ds_ranked, BS_SAMPLE, ev_shape=EV_SHAPE)
        print(ds_gev)
        to_zarr(ds_gev, config.path_gev)

        ## GoF ##
        print('## Goodness of fit: {} ##'.format(datetime.now()))
        ds_gev = xr.open_zarr(config.path_gev)
        ds_gof = step4_goodness_of_fit(ds_gev, chunk_size, ev_shape=EV_SHAPE)
        print(ds_gof)
        to_zarr(ds_gof, config.path_gof)


if __name__ == "__main__":
    sys.exit(main())
