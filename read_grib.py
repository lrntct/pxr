#!/usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function, absolute_import

import sys
import os
from datetime import datetime

import netCDF4
import iris
from cf_units import Unit
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr


GRIB_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_grib'
ZARR_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_zarr'
DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
LONG_NAME = "Total precipitation"
VAR_NAME = 'precipitation'
UNIT = 'm'

YEAR_START = 1979
YEAR_END = 1999

FILE_CONCAT_PREFIX = 'era5_1979-1999_precip'

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

def list_gribs(years):
    grib_list = []
    for year in years:
        for month in [str(m).zfill(2) for m in range(1, 13)]:
            file_name = '{}-{}.grib'.format(year, month)
            grib_list.append(os.path.join(GRIB_DIR, file_name))
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


# def compare_nc_encoding():
#     filename = '2000-01.grib'
#     filepath = os.path.join(DATA_DIR, filename)
#     # load with iris
#     cube = iris.load_cube(filepath)
#     sanitise_cube(cube)
#     # to xarray
#     chunks = {'time': -1, 'latitude': 'auto', 'longitude': 'auto'}
#     dc = xr.DataArray.from_iris(cube).chunk(chunks)
#
#     # to mm/hr
#     dc_hr = dc*3600
#     print(dc_hr)
#
#     # save to netcdf
#     nc_file_int16 = os.path.join(DATA_DIR, 'int16.nc')
#     nc_file_float32 = os.path.join(DATA_DIR, 'float32.nc')
#     dc_hr.to_netcdf(nc_file_float32, encoding=encoding_float32)
#     dc_hr.to_netcdf(nc_file_int16, encoding=encoding_int16)
#     # read from netcdf
#     dc_int16 = xr.open_dataset(nc_file_int16, chunks=chunks).precipitation
#     dc_float32 = xr.open_dataset(nc_file_float32, chunks=chunks).precipitation
#     # Compute errors
#     dc_int16_mae = xr.ufuncs.fabs(dc_int16 - dc_hr).mean()
#     dc_int16_rmse = xr.ufuncs.sqrt(dc_int16 - dc_hr).mean()
#     dc_float32_mae = xr.ufuncs.fabs(dc_float32 - dc_hr).mean()
#     dc_float32_rmse = xr.ufuncs.sqrt(dc_float32 - dc_hr).mean()
#     # Print results
#     print(dc_int16)
#     print('<int16> MAE: {}, RMSE: {}'.format(dc_int16_mae.values, dc_int16_rmse.values))
#     print('<float32> MAE: {}, RMSE: {}'.format(dc_float32_mae.values, dc_float32_rmse.values))


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


def gribs2zarr(years):
    for grib in list_gribs(years):
        print(grib)
        grib2zarr(grib, ZARR_DIR)


def xarray_concat_zarr(zarr_path, years):
    zarr_stores = [f for f in os.listdir(zarr_path) if f.endswith('.zarr')]
    da_list = [xr.open_zarr(os.path.join(zarr_path, f))
               for f in zarr_stores if int(f[:4]) in years]
    da_full = xr.auto_combine(da_list).chunk(ZARR_CHUNKS)
    print(da_full)
    out_path = os.path.join(DATA_DIR, '{}.zarr'.format(FILE_CONCAT_PREFIX))
    print(out_path)
    da_full.to_zarr(out_path, mode='w', encoding=ZARR_ENCODING)


def main():
    # compare_zarr()
    # compare_grib()
    years = range(YEAR_START, YEAR_END+1)


    with ProgressBar():
        # gribs2zarr(years)
        xarray_concat_zarr(ZARR_DIR, years)

#     grib_path = os.path.join(GRIB_DIR, '1989-11.grib')
#     print(grib_path)
#     cube = iris.load_cube('/home/lunet/gylc4/geodata/ERA5/ensemble/1979-09.grib')
#     print(cube)


if __name__ == "__main__":
    sys.exit(main())
