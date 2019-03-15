#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function, absolute_import

import sys
import os
from datetime import datetime

from cf_units import Unit
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import zarr


GRIB_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_grib'
ZARR_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_zarr'
DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
LONG_NAME = "Total precipitation"
VAR_NAME = 'precipitation'
UNIT = 'm'

YEAR_START = 1989
YEAR_END = 1998

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
    grib_list = []
    for year in years:
        for month in [str(m).zfill(2) for m in range(1, 13)]:
            file_name = '{}-{}.{}'.format(year, month, extension)
            grib_list.append(os.path.join(files_dir, file_name))
    return grib_list


def sanitise_cube(cube):
    """Take as an input an iris cube from a CDS ERA5 grib
    remove unecessary coordinates
    set variable name and unit type
    add var_name to coordinates
    """
    cube.remove_coord('forecast_period')
    # cube.remove_coord('originating_centre')
    cube.units = Unit(UNIT)
    cube.long_name = LONG_NAME
    cube.var_name = VAR_NAME
    for coord in cube.coords():
        coord.var_name = coord.name()


def compare_grib():
    filename_1 = '2000-01.grib'
    filename_2 = '2000-01.grib2'
    filepath_1 = os.path.join(DATA_DIR, filename_1)
    filepath_2 = os.path.join(DATA_DIR, filename_2)
    cube_1 = iris.load_cube(filepath_1)
    cube_2 = iris.load_cube(filepath_2)
    sanitise_cube(cube_1)
    sanitise_cube(cube_2)
    chunks = {'time': -1, 'latitude': 'auto', 'longitude': 'auto'}
    da_1 = xr.DataArray.from_iris(cube_1).chunk(chunks)
    da_2 = xr.DataArray.from_iris(cube_2).chunk(chunks)
    mae = xr.ufuncs.fabs(da_1 - da_2).mean()
    rmse = xr.ufuncs.sqrt(da_1 - da_2).mean()
    print('MAE: {}, RMSE: {}'.format(mae.values, rmse.values))


def compare_zarr():
    filename = '2000-01.grib'
    filepath = os.path.join(DATA_DIR, filename)
    time_dict = {}
    # load with iris
    time_iris_load = datetime.now()
    cube = iris.load_cube(filepath)
    time_dict['iris_load'] = datetime.now()- time_iris_load
    sanitise_cube(cube)
    # to xarray
    chunks = {'time': -1, 'latitude': 'auto', 'longitude': 'auto'}
    dc = xr.DataArray.from_iris(cube).chunk(chunks)
    # to mm/hr
    time_iris_tohr = datetime.now()
    dc_hr = dc*3600.
    time_dict['iris_tohr'] = datetime.now() - time_iris_tohr

    for encoding_name, encoding in encodings.items():
        for comp_name, compressor in compressors.items():
            file_name = '{}_{}'.format(encoding_name, comp_name)
            print(file_name)
            encoding[VAR_NAME]['compressor'] = compressor
            # save to zarr
            zarr_file = os.path.join(DATA_DIR, '{}.zarr'.format(file_name))
            time_2disk = datetime.now()
            dc_hr.to_dataset().to_zarr(zarr_file, encoding=encoding)
            time_dict['{}_2disk'.format(file_name)] = datetime.now() - time_2disk
            # read from zarr
            time_comp = datetime.now()
            dc_zarr = xr.open_zarr(zarr_file).precipitation
            # Compute errors
            mae = xr.ufuncs.fabs(dc_zarr - dc_hr).mean()
            rmse = xr.ufuncs.sqrt(dc_zarr - dc_hr).mean()
            # Print results
            print('<{}> MAE: {}, RMSE: {}'.format(file_name, mae.values, rmse.values))
            time_dict['{}_comp'.format(file_name)] = datetime.now() - time_comp

    for s, t in time_dict.items():
        print('{}: {}'.format(s, t))


def grib2zarr(grib_path, zarr_dir):
    """transform grib to zarr
    """
    # load grib as xarray
    cube = iris.load_cube(grib_path)
    sanitise_cube(cube)
    da = xr.DataArray.from_iris(cube).chunk(ZARR_CHUNKS)
    # mm/hr
    # da_hr = da * 3600.  # Mean rate
    da_hr = da * 1000.  # Total precip
    basename = os.path.splitext(os.path.basename(grib_path))[0]
    zarr_filename = '{}.zarr'.format(basename)
    out_file_path = os.path.join(zarr_dir, zarr_filename)
    print(out_file_path)
    da_hr.to_dataset().to_zarr(out_file_path, mode='w', encoding=ZARR_ENCODING)


def netcdf2zarr(nc_path, zarr_dir, mult):
    ds = xr.open_dataset(nc_path).chunk(ZARR_CHUNKS)
    ds_hr = ds * mult
    ds_hr = ds_hr.rename({'tp': VAR_NAME})
    basename = os.path.splitext(os.path.basename(nc_path))[0]
    zarr_filename = '{}.zarr'.format(basename)
    out_file_path = os.path.join(zarr_dir, zarr_filename)
    print(out_file_path)
    print(ds_hr)
    ds_hr.to_zarr(out_file_path, mode='w', encoding=ZARR_ENCODING)


def files2zarr(years, files_dir, extension, multiplier):
    for f in list_files(years, files_dir, extension):
        print(f)
        if extension == 'grib':
            grib2zarr(f, ZARR_DIR)
        elif extension == 'netcdf':
            netcdf2zarr(f, ZARR_DIR, multiplier)


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


def merge_per_year(year, out_path):
    ds_list = []
    for filename in os.listdir(ZARR_DIR):
        if str(year) in filename:
            file_path = os.path.join(ZARR_DIR, filename)
            ds = xr.open_zarr(file_path)
            ds_list.append(ds)
    ds_all = xr.concat(ds_list, dim='time').sortby('time').chunk(ZARR_CHUNKS)
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
        # files2zarr(years, '/home/lunet/gylc4/geodata/ERA5/ensemble', 'netcdf', 10000)
        xarray_concat_zarr(ZARR_DIR, years)
        # merge_per_year(1989, os.parth.join(DATA_DIR, 'era5_1989_precip.zarr'))
        # update_precip('era5_1979-2018_precip.zarr', 'era5_1989_precip.zarr')



if __name__ == "__main__":
    cluster = LocalCluster(n_workers=32, threads_per_worker=1)
    print(cluster)
    client = Client(cluster)
    sys.exit(main())
