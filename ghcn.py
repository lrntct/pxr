# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import os, sys
import itertools
import copy

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
import pandas as pd
import cartopy as ctpy
import matplotlib.pyplot as plt

# df_stations = ulmo.ncdc.ghcn_daily.get_stations(start_year=2000, end_year=2013, as_dataframe=True)
# print(df_stations)
# print(len(df_stations))

# df_data = ulmo.ncdc.ghcn_daily.get_data(station_id='MX000766800', as_dataframe=True)
# print(df_data)

GHCN_DIR = '../data/GHCN/'
STATION_FILE = 'ghcnd-stations.txt'
STATIONS_COLSPEC = [(0,12), (12,21), (21,31), (31,38), (38,41), (41,72), (72,76), (76,80), (80,86)]
COL_DATA = ['ID', 'DATE', 'OBS_TYPE', 'OBS_VALUE', 'MFLAG', 'QFLAG', 'SFLAG', 'OBS_TIME']
COL_STATIONS = ['ID', "LATITUDE","LONGITUDE", 'ELEVATION', 'STATE', 'NAME', 'GSN_FLAG', 'HCN_FLAG', 'WMO_ID']
CHUNKS = {'time': -1, 'station': 500}

GHCN_ALL = 'ghcn2000-2017.zarr'
GHCN_SELECT = 'ghcn2000-2017_select.zarr'
GHCN_FILTERED = 'ghcn2000-2017_filtered.nc'

# Standard deviation threshold for weekday bias detection
WEEKDAY_STD_THRES = 0.1
# Underreporting bias threshold in inches (from Daly & Al. 2007), converted to mm/hour
UNDERREP_IN = {'high': (1.52/24, 2.54/24), 'low': (0.25/24, 1.27/24)}
# Equivalent for measurements in mm
UNDERREP_MM = {'high': (1.1/24, 3/24), 'low': (0.1/24, 1./24)}
# Underreporting bias threshold
UNDERREP_THRES = 0.6

def ghcn_read_data(data_dir, df_stations_infos):
    # read all CSV in dir and put them in a list of DataFrame
    df_list = []
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            print(f)
            f_path = os.path.join(data_dir, f)
            df = pd.read_csv(f_path, parse_dates=['DATE'], names=COL_DATA, header=None,
                             usecols=['ID', 'DATE', 'OBS_TYPE', 'OBS_VALUE', 'QFLAG'],
                            #  nrows=1000
                             )
            # keep only the precipitation
            df = (df[df['OBS_TYPE'] == 'PRCP'].drop(columns='OBS_TYPE')
                                              .rename(columns={'OBS_VALUE': 'precipitation',
                                                               'DATE': 'time'}))
            # Keep only the row without any quality flag
            df_list.append(df[df['QFLAG'].isnull()])
        # break
    # All years and stations in one DataFrame
    df_all = pd.concat(df_list)
    # Split by station
    gb = df_all.groupby('ID')
    df_databystations = [copy.deepcopy(gb.get_group(x)) for x in gb.groups]

    # Create a xarray DataArray
    stations = []
    # idx = [[] for i in range(len(COL_NAMES))]
    for df in df_databystations:
        station_id = str(df['ID'].unique()[0])
        df.set_index('time', drop=True, inplace=True)
        # We want mm/h, not per day
        df.loc[:, 'precipitation'] = df['precipitation'] / 24.
        # Keep only the column with precipitation
        da = df['precipitation'].to_xarray()
        da = da.expand_dims('station')
        station_infos = df_stations_infos.loc[station_id]
        try:
            da.coords['id'] = xr.DataArray([station_id], dims='station')
            da.coords['name'] = xr.DataArray([station_infos['NAME']], dims='station')
            da.coords['latitude'] = xr.DataArray([station_infos['LATITUDE']], dims='station',)
            da.coords['longitude'] = xr.DataArray([station_infos['LONGITUDE']], dims='station')
            da.coords['elevation'] = xr.DataArray([station_infos['ELEVATION']], dims='station')
        except IndexError:
            pass
            print(df)
        else:
            stations.append(da)
        # break

    da_concat = xr.concat(stations, dim='station')
    # date_idx = da_concat.coords['date'].values
    # # Create multi-index for the other coordiantes (share the dimension)
    # midx = pd.MultiIndex.from_arrays(idx, names=('code', 'name', 'latitude', 'longitude'))
    # da = xr.DataArray(da_concat.values, name='precipitation',
    #                   coords={'station': midx, 'time':date_idx}, dims=['station', 'time'])
    return da_concat.to_dataset()


def drop_stations(ds):
    """keep only stations with more than 90% years with more than 90% of days
    """
    # Add a unique int id to each station
    ds.coords['int_id'] = xr.DataArray(np.arange(len(ds['station'])), dims='station',)
    min_days = int(365*.9)
    # Number of values per year
    da_year_count = ds['precipitation'].groupby('time.year').count(dim='time')
    min_years = int(len(da_year_count['year']) * .9)
    # Number of years with more records than the threshold
    da_num_of_full_years = (da_year_count
                                .where(da_year_count > min_days)
                                .count(dim='year')
                                .rename('num_of_full_years'))
    ds['full_years'] = da_num_of_full_years
    # Selection of stations with more full years than the threshold
    da_full_years_per_station = da_num_of_full_years.where(da_num_of_full_years > min_years, drop=True)
    # Array of stations indices with more full years than the threshold
    kept_station_int_id = da_full_years_per_station['int_id'].values
    # Return an extract of the original data
    return ds.loc[{'station':kept_station_int_id}]


def bias_flag(ds):
    """
    """
    # Day of the week bias
    weekday_mean = ds['precipitation'].groupby('time.dayofweek').mean(dim='time')
    ds['dayofweek_std'] = weekday_mean.std(dim='dayofweek')
    # ds['dayofweek_mean'] = weekday_mean.mean(dim='dayofweek')

    # Underreporting
    for name, th in zip(['urep_bias_in', 'urep_bias_mm'], [UNDERREP_IN, UNDERREP_MM]):
        # Select the values
        l_indexer = np.less_equal(th['low'][0], ds['precipitation']) & np.less_equal(ds['precipitation'], th['low'][1])
        h_indexer = np.less_equal(th['high'][0], ds['precipitation']) & np.less_equal(ds['precipitation'], th['high'][1])
        # Count them
        rept_lower = xr.where(l_indexer, True, False).sum(dim='time').astype(np.float32)
        rept_higher = xr.where(h_indexer, True, False).sum(dim='time').astype(np.float32)
        # Calculate the underreporting bias (Daly & Al. 2007)
        ds[name] = np.divide(rept_higher, rept_lower, out=rept_higher.values, where=rept_lower.values>0)

    # Select the station that pass the repporting bias tests
    urep_indexer = np.logical_and(ds['urep_bias_in'] < UNDERREP_THRES,
                                 ds['urep_bias_mm'] < UNDERREP_THRES)
    sel_indexer = np.logical_and(ds['dayofweek_std'] < WEEKDAY_STD_THRES,
                                urep_indexer)
    ds_sel = ds.where(sel_indexer, drop=True)
    return ds_sel


def main():
    with ProgressBar():
        # df_stations_infos = pd.read_fwf(os.path.join(GHCN_DIR, STATION_FILE),
        #                           names=COL_STATIONS, index_col='ID', header=None, colspecs=STATIONS_COLSPEC)
        # ds = ghcn_read_data('/home/lunet/gylc4/geodata/GHCN/', df_stations_infos).chunk(CHUNKS)
        # ds.reset_index('station').to_zarr(os.path.join(GHCN_DIR, GHCN_ALL), mode='w')
        # ds = xr.open_zarr(os.path.join(GHCN_DIR, GHCN_ALL)).chunk(CHUNKS).rename({'station_':'id'})#.isel(station=slice(0, 500))
        # print(ds)
        # print(df_stations_infos.head())
        # keep only stations with enough records
        # ds_select = drop_stations(ds)
        # print(ds_select.load())
        # print()
        # ds_select.to_zarr(os.path.join(GHCN_DIR, GHCN_SELECT), mode='w')

        ds_select = xr.open_zarr(os.path.join(GHCN_DIR, GHCN_SELECT))
        ds_select = bias_flag(ds_select.load())
        ds_select.to_netcdf(os.path.join(GHCN_DIR, GHCN_FILTERED))

        # print(np.isfinite(da).sum(dim='time'))
        # da.plot()
        # plt.savefig('ghcn_test.png')
        # plt.close()


if __name__ == "__main__":
    sys.exit(main())
