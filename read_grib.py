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
# import cfgrib
# from cfgrib import xarray_store
import zarr
# from dask.distributed import Client

import seaborn as sns
import cartopy as ctpy
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from pyinstrument import Profiler

GRIB_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly'
ZARR_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_zarr'
DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
LONG_NAME = "Mean total precipitation rate"
VAR_NAME = 'precipitation'
UNIT = 'kg m-2 s-1'
NUM_CHUNKS = 8
CHUNK_SIZE = 15  # in degree

YEAR_START = 2001
YEAR_END = 2012

FILE_CONCAT_PREFIX = 'era5_precip'

# Coordinates of study sites
KAMPALA = (0.317, 32.616)
KISUMU = (-0.1, 34.75)

DURATION = [3, 6, 12, 24]

# Spatial resolution in degree - used for match coordinates
SPATIAL_RES = 0.25

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


def chunks_list(extent, chunk_size):
    if extent==180:
        start_point = -extent/2
    elif extent==360:
        start_point = 0
    else:
        raise ValueError()
    start_chunks = [start_point + i * CHUNK_SIZE for i in range((extent // chunk_size))]
    return [(float(i), float(i+chunk_size)) for i in start_chunks]


def coor_name(coor, lower, higher):
    if coor < 0:
        return '{}{}'.format(int(abs(coor)), lower)
    else:
        return '{}{}'.format(int(coor), higher)


def out_filename(lat_chunk, lon_chunk):
    w = coor_name(lon_chunk[0], 'w', 'e')
    e = coor_name(lon_chunk[1], 'w', 'e')
    s = coor_name(lat_chunk[0], 's', 'n')
    n = coor_name(lat_chunk[1], 's', 'n')
    return '{}{}{}{}.zarr'.format(w,e,s,n)


def list_gribs(years):
    grib_list = []
    for year in years:
        for month in [str(m).zfill(2) for m in range(1, 13)]:
            file_name = '{}-{}.grib'.format(year, month)
            grib_list.append(os.path.join(GRIB_DIR, file_name))
    return grib_list


def round_partial(value):
    return round(value / SPATIAL_RES) * SPATIAL_RES


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


def plot(da):
    """https://cbrownley.wordpress.com/2018/05/15/visualizing-global-land-temperatures-in-python-with-scrapy-xarray-and-cartopy/
    """
    plt.figure(figsize=(5, 3))
    ax_p = plt.gca(projection=ctpy.crs.Robinson(), aspect='auto')
    ax_p.coastlines(linewidth=.3, color='black')
    da.mean(dim='time').plot.imshow(ax=ax_p, transform=ctpy.crs.PlateCarree(),
                                   extend='max', vmax=20,
                                   cbar_kwargs=dict(orientation='horizontal',
                                                    label='Precipitation rate (mm/hr)'))
    # colorbar
    # cbar = plt.colorbar(temp_plot, orientation='horizontal')
    # cbar.set_label(label='Precipitation rate (mm/hr)')
    plt.title("Mean hourly precipitation in 2000 (ERA5)")
    plt.savefig('max.png')
    plt.close()


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


def grib2zarr(grib_path):
    """transform grib to zarr
    """
    chunks = {'time': -1, 'latitude': 20, 'longitude': 20}
    encoding = encoding_float16
    encoding[VAR_NAME]['compressor'] = zarr.Blosc(cname='lz4', clevel=9)
    # grib_file = '{}.grib'.format(name)
    basename = os.path.splitext(grib_path)[0]
    zarr_filename = '{}.zarr'.format(basename)
    # grib_path = os.path.join(GRIB_DIR, grib_file)
    # load grib as xarray
    cube = iris.load_cube(grib_path)
    sanitise_cube(cube)
    da = xr.DataArray.from_iris(cube).chunk(chunks)
    out_file_path = os.path.join(ZARR_DIR, zarr_filename)
    da.to_dataset().to_zarr(out_file_path, encoding=encoding)


def gribs2zarr(years):
    for grib in list_gribs(years):
        print(grib)
        grib2zarr(grib)


def grib2grib(years):
    """
    """
    for year in years:
        grib_file = '{}.grib'.format(year)
        out_file = '{}.grib2'.format(year)
        grib_path = os.path.join(DATA_DIR, grib_file)
        # load grib as xarray
        cube = iris.load_cube(grib_path)
        sanitise_cube(cube)
        out_file_path = os.path.join(DATA_DIR, out_file)
        iris.save(cube, out_file_path)


def load_grib2(years):
    """
    """
    chunks = {'time': -1, 'latitude': 'auto', 'longitude': 'auto'}
    for year in years:
        grib_file = '{}.grib2'.format(year)
        grib_path = os.path.join(DATA_DIR, grib_file)
        # load grib as xarray
        cube = iris.load_cube(grib_path)
        sanitise_cube(cube)
        da = xr.DataArray.from_iris(cube).chunk(chunks)
        print(da.time.values)


def concat2zarr(years):
    """
    convert to mm/hr
    concatenate
    """
    chunks = {'time': -1, 'latitude': 90, 'longitude': 90}
    encoding = encoding_float16
    encoding[VAR_NAME]['compressor'] = zarr.Blosc(cname='lz4', clevel=9)
    da_list = []
    cube = iris.load_cube(list_gribs(years))
    # print(cubes)
    # cube = cubes.concatenate_cube()
    sanitise_cube(cube)
    print(cube)
    da = xr.DataArray.from_iris(cube)
    print(da)
    out_filename = '{}_{}-{}.nc'.format(FILE_CONCAT_PREFIX, YEAR_START, YEAR_END)
    out_file_path = os.path.join(DATA_DIR, out_filename)
    # iris.save(cube, out_file_path)
    dask.visualize(da, filename='da.svg')
    # da.to_dataset().to_zarr(out_file_path, encoding=encoding)
        # ds = cfgrib.Dataset.frompath(grib_path)
        # da = xr.open_zarr(input_path).precipitation
        # da = xarray_store.open_dataset(input_path).to_array(dim=VAR_NAME).chunk(chunks)
        # print(da)
    #     da_list.append(da*3600.)
    #
    # print("concat...")
    # da_large = xr.concat(da_list, dim='time')

    # da_large.to_dataset().to_zarr(out_file_path, encoding=encoding)


def xarray_concat_zarr(zarr_path):
    chunks = {'time': 365*24, 'latitude': 10, 'longitude': 10}
    encoding = {'precipitation': {'dtype': 'float16', 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                'latitude': {'dtype': 'float32'},
                'longitude': {'dtype': 'float32'}}
    da_list = []
    for zarr_store in os.listdir(zarr_path):
        zarr_store_path = os.path.join(zarr_path, zarr_store)
        da = xr.open_zarr(zarr_store_path)
        da_list.append(da)
    da_full = xr.auto_combine(da_list).chunk(chunks)
    print(da_full)
    out_path = os.path.join(DATA_DIR, '{}.zarr'.format(FILE_CONCAT_PREFIX))
    da_full.to_zarr(out_path, mode='w', encoding=encoding)


def netcdf2zarr(path):
    chunks = {'time': -1, 'lat': 'auto', 'lon': 'auto'}
    encoding = {'mtpr': {'dtype': 'float16', 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                'lat': {'dtype': 'float32'},
                'lon': {'dtype': 'float32'}}
    ds = xr.open_dataset(path, chunks=chunks)
    da_mean = ds.mtpr.mean(dim='time')
    print(ds.mtpr)
    # print(list(ds.lon))
    # print(ds.mtpr.encoding)

    # print(da_mean.compute())
    # for c in ds.mtpr:
    #     print(c)
    # dask.visualize(da_mean, filename='mean.png')

    lon_chunks = chunks_list(360, CHUNK_SIZE)
    lat_chunks = chunks_list(180, CHUNK_SIZE)
    # print(lon_chunks, lat_chunks)
    print('number of chunks: {}'.format(len(lon_chunks)*len(lat_chunks)))
    for lat_chunk in lat_chunks:
        for lon_chunk in lon_chunks:
            print(lat_chunk, lon_chunk)
            ds_part = ds.sel(lat=slice(lat_chunk[1], lat_chunk[0]), lon=slice(*lon_chunk)).chunk(chunks)
            out_path = os.path.join(DATA_DIR, out_filename(lat_chunk, lon_chunk))
            print(out_path)
            print(ds_part)
            # ds_part.to_zarr(out_path, encoding=encoding)
            break
        break


def dask_netcdf(nc_path):
    chunks = {'time': -1, 'lat': 'auto', 'lon': 'auto'}
    encoding = {'mtpr': {'dtype': 'float16', 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                'lat': {'dtype': 'float32'},
                'lon': {'dtype': 'float32'}}
    nc_store = netCDF4.Dataset(nc_path, "r", format="NETCDF4")
    da = dask.array.from_array(nc_store, chunks=chunks)
    print(da)


def compare_chunks(nc_path):
    profiler = Profiler()
    chunks_lst = [{'lon': 30}
                  ]
    for chunks in chunks_lst:
        print(chunks)
        profiler.start()
        with xr.open_dataset(nc_path, chunks=chunks) as ds:
            print(ds.mean(dim='time').load())
            # print(ds)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))


def xarray_roll_win(zarr_path):
    ds = xr.open_zarr(zarr_path)
    print(ds)
    ds_mean = ds.mean()
    print(ds_mean)


def main():
    # compare_zarr()
    # compare_grib()
    years = range(YEAR_START, YEAR_END+1)
    # print('grib2grib')
    # grib2grib(years)
    # print('grib2zarr')

    # rolling window analysis - 3h max
    # cube_agreg = cube.rolling_window('time', iris.analysis.MAX, 3)
    # print(cube_agreg)
    # client = Client()
    # client
    with ProgressBar():
        # netcdf2zarr(os.path.join(DATA_DIR, 'era5_precip.nc'))
        # dask_netcdf(os.path.join(DATA_DIR, 'era5_precip_2000.nc'))
        # gribs2zarr(years)
        xarray_concat_zarr(ZARR_DIR)
        # zarr_path = os.path.join(DATA_DIR, 'era5_precip2000.zarr')
        # plot(xr.open_zarr(zarr_path).precipitation)
        # xarray_roll_win(())


    # compare_chunks('/home/laurent/Documents/GeoData/ERA5/era5_precip.nc')

    # concat2netcdf(years)

    # kampala
    # k_lat = round_partial(KAMPALA[0])
    # k_lon = round_partial(KAMPALA[1])
    # Print from the original file
    # series_kampala_diff = dc_diff.sel(**{'latitude':k_lat, 'longitude':k_lon})




if __name__ == "__main__":
    sys.exit(main())
