#!/usr/bin/env python
# -*- coding: utf8 -*-

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

import sys
import os
from datetime import datetime

import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import zarr


ZARR_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_zarr'
YEARLY_DIR = '/home/lunet/gylc4/geodata/ERA5/yearly_zarr'
DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
LONG_NAME = "Total precipitation"
VAR_NAME = 'precipitation'

YEAR_START = 1989
YEAR_END = 2018

FILE_CONCAT_PREFIX = 'era5_{}-{}_precip'.format(YEAR_START, YEAR_END)

ZARR_CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}  # 4: 1deg
ZARR_ENCODING = {'precipitation': {'dtype': 'float32', 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                 'latitude': {'dtype': 'float32'},
                 'longitude': {'dtype': 'float32'}}

# storage encoding
encoding_float32 = {VAR_NAME: {'dtype': 'float32'},
                    'latitude': {'dtype': 'float32'},
                    'longitude': {'dtype': 'float32'}}
encoding_float16 = {VAR_NAME: {'dtype': 'float16'},
                    'latitude': {'dtype': 'float32'},
                    'longitude': {'dtype': 'float32'}}
encoding_int16 = {VAR_NAME: {'dtype': 'int16', 'scale_factor': 0.1,'_FillValue': -9999},
                  'latitude': {'dtype': 'float32'},
                  'longitude': {'dtype': 'float32'}}

encodings = {'float32': encoding_float32,
             'float16': encoding_float16}

compressors = {'uncompressed': None,
               'zstd': zarr.Blosc(cname='zstd'),
               'lz4': zarr.Blosc(cname='lz4', clevel=9),
               'lz4hc': zarr.Blosc(cname='lz4hc', clevel=9)}

def list_files(years, files_dir, extension):
    file_list = []
    for year in years:
        for month in [str(m).zfill(2) for m in range(1, 13)]:
            file_name = '{}-{}.{}'.format(year, month, extension)
            file_list.append(os.path.join(files_dir, file_name))
    return file_list


def xarray_concat_zarr(zarr_path, years):
    zarr_stores = [f for f in os.listdir(zarr_path) if f.endswith('.zarr')]
    da_list = []
    for f in zarr_stores:
        if int(f[:4]) in years:
            da = xr.open_zarr(os.path.join(zarr_path, f))['precipitation']
            try:
                da = da.drop('originating_centre')
            except ValueError:
                pass
            da_list.append(da)
    da_full = xr.concat(da_list, dim='time').chunk(ZARR_CHUNKS).sortby('time')
    print(da_full)
    out_path = os.path.join(DATA_DIR, '{}.zarr'.format(FILE_CONCAT_PREFIX))
    print(out_path)
    da_full.to_dataset().chunk(ZARR_CHUNKS).to_zarr(out_path, mode='w', encoding=ZARR_ENCODING)


def merge_per_year(years):
    """
    """
    for year in years:
        file_name = '{}.zarr'.format(year)
        file_path = os.path.join(YEARLY_DIR, file_name)
        merge_one_year(year, file_path)


def merge_one_year(year, out_path):
    """Merge zarr files into a single year_file
    """
    ds_list = []
    for filename in os.listdir(ZARR_DIR):
        if str(year) in filename:
            file_path = os.path.join(ZARR_DIR, filename)
            ds = xr.open_zarr(file_path)
            try:
                ds = ds.drop('originating_centre')
            except ValueError:
                pass
            ds_list.append(ds)
    ds_all = xr.concat(ds_list, dim='time').sortby('time').chunk(ZARR_CHUNKS)
    print(ds_all)
    ds_all.to_zarr(out_path, mode='w', encoding=ZARR_ENCODING)


def update_precip(main_file, update_file):
    main_path = os.path.join(DATA_DIR, main_file)
    update_path = os.path.join(DATA_DIR, update_file)
    da_main = xr.open_zarr(main_path)['precipitation']
    da_update = xr.open_zarr(update_path)['precipitation']
    da_updated = da_main.combine_first(da_update)
    print(da_updated)
    nan_count = np.isnan(da_updated).sum()
    print(nan_count.load())



def main():
    # compare_zarr()
    # compare_grib()
    years = range(YEAR_START, YEAR_END+1)

    with ProgressBar():
        # ds = xr.open_zarr(os.path.join(DATA_DIR, 'era5_1989-1998_precip.zarr'))
        # print(ds)
        # xarray_concat_zarr(ZARR_DIR, years)
        merge_per_year(years)
        # merge_per_year(1989, os.parth.join(DATA_DIR, 'era5_1989_precip.zarr'))
        # update_precip('era5_1979-2018_precip.zarr', 'era5_1989_precip.zarr')



if __name__ == "__main__":
    cluster = LocalCluster(n_workers=32, threads_per_worker=1)
    print(cluster)
    client = Client(cluster)
    sys.exit(main())
