# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr
import scipy

import seaborn as sns
import cartopy as ctpy
import matplotlib.pyplot as plt

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_precip_big_chunks.zarr'
ANNUAL_FILE = 'era5_precip_annual_max.zarr'
ANNUAL_FILE_GUMBEL = 'era5_precip_gumbel.nc'

# Coordinates of study sites
KAMPALA = (0.317, 32.616)
KISUMU = (0.1, 34.75)
# Extract
EXTRACT = dict(latitude=slice(1.0, -0.25),
               longitude=slice(32.5, 35))
# EXTRACT = dict(latitude=slice(45, -45),
#                longitude=slice(0, 90))
# Spatial resolution in degree - used for match coordinates
SPATIAL_RES = 0.25

# Event duration in hours - has to be adjusted to temporal resolution
DURATIONS = [3, 6, 12, 24]
TEMP_RES = 1  # Temporal resolution in hours

HOURLY_CHUNKS = {'time': 365*24, 'latitude': 8, 'longitude': 8}
ANNUAL_CHUNKS = {'year': -1, 'duration':1, 'latitude': 30*4, 'longitude': 30*4}  # 4 cells: 1 degree
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


def annual_linregress(ds, x, y, prefix):
    """ds: xarray dataset
    x, y: name of variables to use for the regression
    prefix: to be added before the indivudual result names
    """
    # add empty arrays to store results of the regression
    res_shape = tuple(v for k,v in ds[x].sizes.items() if k != 'year')
    res_dims = tuple(k for k,v in ds[x].sizes.items() if k != 'year')
    for res_name in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']:
        arr_name = '{}_{}'.format(prefix, res_name)
        ds[arr_name] = (res_dims, np.zeros(res_shape, dtype='float32'))
    for lat in ds.coords['latitude']:
        for lon in ds.coords['longitude']:
            for duration in ds.coords['duration']:
                locator = {'longitude':lon, 'latitude':lat, 'duration':duration}
                sel = ds.loc[locator]
                res = scipy.stats.linregress(sel[x], sel[y])
                for k, v in res._asdict().items():
                    arr_name = '{}_{}'.format(prefix, k)
                    ds[arr_name].loc[locator] = v


def double_log(arr):
    return (np.log(np.log(1/arr))).astype('float32')


def step2_gumbel_fit(annual_maxs):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    # Rank the observations in time
    ranks = annual_maxs.load().rank(dim='year').rename('rank').astype('int16')
    ds = xr.merge([annual_maxs, ranks])
    # Estimate probability F{x} with plotting positions
    n_obs = ds.annual_max.count(dim='year')
    ds['plot_pos'] = (ds['rank'] / (n_obs+1)).astype('float32')
    ds['gumbel_prov'] = double_log(ds['plot_pos'])
    # First fit
    annual_linregress(ds, 'annual_max', 'gumbel_prov', 'prov_lg')
    # get provisional gumbel parameters
    ds['loc_prov'] = -ds['prov_lg_intercept']/ds['prov_lg_slope']
    ds['scale_prov'] = -1/ds['prov_lg_slope']
    # Analytic probability F(x) from Gumbel CDF
    z = (ds['annual_max'] - ds['loc_prov']) / ds['scale_prov']
    ds['gumbel_cdf'] = np.e**(-np.e**-z)
    # Get the final location and scale parameters
    ds['gumbel_final'] = double_log(ds['gumbel_cdf'])
    annual_linregress(ds, 'annual_max', 'gumbel_final', 'final_lg')
    ds['loc_final'] = -ds['final_lg_intercept']/ds['final_lg_slope']
    ds['scale_final'] = -1/ds['final_lg_slope']

    # save to disk
    out_path = os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL)
    chunked = ds.chunk(ANNUAL_CHUNKS)
    chunked.to_netcdf(out_path, mode='w')


def benchmark(ds):
    """Run the gumbel fit for a number of extract sizes
    print result to stdout
    """
    duration_list = []
    for i in range(5):
        degrees = (i+1)*5
        locator = dict(latitude=slice(degrees, 0),  # Latitudes are in descending order
                       longitude=slice(0, degrees))
        sel = ds.loc[locator]
        start = datetime.datetime.now()
        step2_gumbel_fit(sel)
        duration = datetime.datetime.now() - start
        duration_list.append(duration)
    print(duration_list)

def main():
    kampala_locator = {'latitude': round_partial(KAMPALA[0]),
                       'longitude': round_partial(KAMPALA[1]),
                       'duration': 6}
    with ProgressBar():
        # hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
        # hourly = xr.open_zarr(hourly_path).chunk(HOURLY_CHUNKS)
        # kampala_hourly = hourly.loc[KAMPALA]
        # step1_write_annual_maxs()
        annual_maxs = step1bis_reorg_ds()
        da_kampala = annual_maxs.loc[kampala_locator]
        an_max_extract = annual_maxs.loc[EXTRACT]
        # print(da_extract.load())
        # print(kampala_hourly)
        step2_gumbel_fit(an_max_extract)
        # benchmark(annual_maxs)

        # kamp_6 = ds.loc[kampala_locator]
        # plt.figure(figsize=(8, 5))
        # ax = plt.gca()
        # # # kamp_6.plot_pos.plot(ax=ax)
        # plt.scatter(kamp_6.annual_max, kamp_6.gumbel_final)
        # x1 = np.linspace(np.min(kamp_6.annual_max), np.max(kamp_6.annual_max), 500)
        # fit_line = kamp_6.slope2 * kamp_6.annual_max + kamp_6.intercept2
        # plt.plot(kamp_6.annual_max, fit_line, '-r')
        # # # plt.scatter(kamp_6.annual_max, kamp_6.plot_pos)
        # plt.savefig('kampala.png')
        # plt.close()

        # step2_gumbel_params(da_extract)
        # plot(da.mean(dim='time'))


if __name__ == "__main__":
    sys.exit(main())
