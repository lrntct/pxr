# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
from datetime import datetime

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr
import scipy.optimize

import seaborn as sns
import cartopy as ctpy
import matplotlib.pyplot as plt

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_precip_big_chunks.zarr'
ANNUAL_FILE = 'era5_precip_annual_max.zarr'

# Coordinates of study sites
KAMPALA = (0.317, 32.616)
KISUMU = (-0.1, 34.75)
# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32, 35))
EXTRACT = dict(latitude=slice(45, -45),
               longitude=slice(0, 90))
# Spatial resolution in degree - used for match coordinates
SPATIAL_RES = 0.25

# Event duration in hours - has to be adjusted to temporal resolution
DURATIONS = [3, 6, 12, 24]
TEMP_RES = 1  # Temporal resolution in hours

HOURLY_CHUNKS = {'time': 365*24, 'latitude': 8, 'longitude': 8}
ANNUAL_CHUNKS = {'year': -1, 'latitude': 30*4, 'longitude': 30*4}  # 4 cells: 1 degree
ANNUAL_ENCODING = {'precipitation': {'dtype': 'float32', 'compressor': zarr.Blosc(cname='lz4', clevel=9)},
                   'latitude': {'dtype': 'float32'},
                   'longitude': {'dtype': 'float32'}}

def round_partial(value):
    return round(value / SPATIAL_RES) * SPATIAL_RES


def plot_mean(da):
    """https://cbrownley.wordpress.com/2018/05/15/visualizing-global-land-temperatures-in-python-with-scrapy-xarray-and-cartopy/
    """
    plt.figure(figsize=(8, 5))
    ax_p = plt.gca(projection=ctpy.crs.Robinson(), aspect='auto')
    ax_p.coastlines(linewidth=.3, color='black')
    da.plot.imshow(ax=ax_p, transform=ctpy.crs.PlateCarree(),
                   extend='max', vmax=20,
                   cbar_kwargs=dict(orientation='horizontal', label='Precipitation rate (mm/hr)'))
    # colorbar
    # cbar = plt.colorbar(temp_plot, orientation='horizontal')
    # cbar.set_label(label='Precipitation rate (mm/hr)')
    plt.title("Mean hourly precipitation 2000-2012 (ERA5)")
    plt.savefig('mean.png')
    plt.close()


def annual_maxs_of_roll_mean(ds, durations, temp_res):
    """for each rolling winfows size:
    compute the annual maximum of a moving mean
    return a dataset with the durations as variables
    """
    annual_maxs = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip_roll_mean = ds.precipitation.rolling(time=window_size).mean(dim='time')
        annual_max = precip_roll_mean.groupby('time.year').max(dim='time')
        annual_max.name = 'annual_max_{}h'.format(duration)
        annual_maxs.append(annual_max)
    return xr.merge(annual_maxs).chunk(ANNUAL_CHUNKS)


def step1_write_annual_maxs():
    hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
    hourly = xr.open_zarr(hourly_path).chunk(HOURLY_CHUNKS)
    annual_maxs = annual_maxs_of_roll_mean(hourly, DURATIONS, TEMP_RES)
    out_path = os.path.join(DATA_DIR, ANNUAL_FILE)
    annual_maxs.to_zarr(out_path, mode='w')


def step1bis_reorg_ds():
    """take a dataset as an imput
    Re-arrange the variables as member of a new dimension
    retrurn an xarray
    """
    annual_path = os.path.join(DATA_DIR, ANNUAL_FILE)
    annual_maxs = xr.open_zarr(annual_path)
    an_max_list = []
    for dur in DURATIONS:
        var_name = 'annual_max_{}h'.format(dur)
        da = annual_maxs[var_name].rename('annual_max').expand_dims('duration')
        da.coords['duration'] = [dur]
        an_max_list.append(da)
    da_full = xr.concat(an_max_list, 'duration')
    return da_full


def gumbel_pdf(x, loc, scale):
    """Returns the value of Gumbel's pdf with parameters loc and scale at x.
    https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize
    """
    # substitute
    z = (x - loc)/scale
    return (1./scale) * (np.exp(-(z + (np.exp(-z)))))


def step2_fit_gumbel(annual_maxs):
    # ranks = annual_maxs.load().rank(dim='year').rename('rank')
    # ds = xr.merge([ranks, annual_maxs])
    # print(ds)
    # https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize
    f = lambda p, x: (-np.log(gumbel_pdf(x, p[0], p[1]))).sum()
    gumbel_params = scipy.optimize.fmin(f, [0.5,0.5], args=(annual_maxs.values,))
    print(gumbel_params)
    # annual_maxs.plot()
    # plt.savefig('annual_max.png')


def main():
    # location of kampala
    k_lat = round_partial(KAMPALA[0])
    k_lon = round_partial(KAMPALA[1])
    with ProgressBar():
        # step1_write_annual_maxs()
        annual_maxs = step1bis_reorg_ds()
        da_kampala = annual_maxs.loc[{'latitude':k_lat, 'longitude':k_lon, 'duration':6}]

        step2_fit_gumbel(da_kampala)
        # plot(da.mean(dim='time'))


if __name__ == "__main__":
    sys.exit(main())
