# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import os, sys
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import ulmo
import pandas as pd
import cartopy as ctpy
import matplotlib.pyplot as plt

# df_stations = ulmo.ncdc.ghcn_daily.get_stations(start_year=2000, end_year=2013, as_dataframe=True)
# print(df_stations)
# print(len(df_stations))

# df_data = ulmo.ncdc.ghcn_daily.get_data(station_id='MX000766800', as_dataframe=True)
# print(df_data)

GHCN_DIR = '../data/GHCN/'

COL_NAMES = ["STATION","NAME","LATITUDE","LONGITUDE"]


def ghcn_read_csv(data_dir):
    df_list = []
    # read all CSV in dir and put them in a list of DataFrame
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            f_path = os.path.join(data_dir, f)
            df = pd.read_csv(f_path, index_col='DATE', parse_dates=True,
                             usecols=COL_NAMES + ["DATE","PRCP"])
            df_list.append(df)
    # All stations in one DataFrame
    df_all = pd.concat(df_list)
    df_all.index.names = ['date']
    # Split by station
    gb = df_all.groupby('STATION')
    df_stations = [gb.get_group(x) for x in gb.groups]

    # Create a xarray DataArray
    stations = []
    idx = [[] for i in range(len(COL_NAMES))]
    for df in df_stations:
        # delete duplicated index
        df_dropped = df[~df.index.duplicated(keep='first')]
        # Populate index
        for i, col_name in enumerate(COL_NAMES):
            idx[i].append(df_dropped[col_name][0])
        # GHCN is in tenth of mm/day, we want mm/h
        da = df_dropped.PRCP.to_xarray() / 24.
        da.name = 'precipitation'
        da = da.expand_dims('station')
        da.coords['station'] = [df_dropped['STATION'].unique()[0]]
        stations.append(da)

    da_concat = xr.concat(stations, dim='station')
    date_idx = da_concat.coords['date'].values
    # Create multi-index for the other coordiantes (share the dimension)
    midx = pd.MultiIndex.from_arrays(idx, names=('code', 'name', 'latitude', 'longitude'))
    da = xr.DataArray(da_concat.values, name='precipitation',
                      coords={'station': midx, 'time':date_idx}, dims=['station', 'time'])
    return da.to_dataset()


def drop_stations(ds, ymin, ymax):
    """keep only stations with more than 90% years with more than 90% of days
    """
    min_days = int(365*.9)
    da_year_count = ds['precipitation'].groupby('time.year').count(dim='time')
    year_count_sel = da_year_count.sel(year=slice(ymin, ymax))
    min_years = int(len(year_count_sel['year']) * .9)
    num_of_full_years = year_count_sel.where(year_count_sel > min_days, drop=True).count(dim='year')
    kept_stations = num_of_full_years.where(num_of_full_years > min_years, drop=True)['code'].values
    all_stations = da_year_count['code'].values
    drop_stations = [s for s in all_stations if s not in kept_stations]
    return ds.drop(drop_stations, dim='station').sel(time=slice(str(ymin), str(ymax)))


def main():
    ds = ghcn_read_csv(GHCN_DIR)
    # ds.reset_index('station').to_netcdf(os.path.join(GHCN_DIR, 'ghcn.nc'), mode='w')

    # keep only stations with enough records
    ds_cleaned = drop_stations(ds, 2000, 2012)
    print(ds_cleaned)
    ds_cleaned.reset_index('station').to_zarr(os.path.join(GHCN_DIR, 'ghcn_2000-2012_precip.zarr'), mode='w')

    # print(np.isfinite(da).sum(dim='time'))
    # da.plot()
    # plt.savefig('ghcn_test.png')
    # plt.close()


if __name__ == "__main__":
    sys.exit(main())
