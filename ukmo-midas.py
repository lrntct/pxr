# -*- coding: utf8 -*-
import sys
import os
import io
import copy
import tarfile
import gzip
import shutil
import datetime
import multiprocessing as mp

import requests
import xarray as xr
import pandas as pd
import dask
from dask.diagnostics import ProgressBar
import zarr
import numpy as np
import shapely.geometry
import geopandas as gpd

START_YEAR = 1979
END_YEAR = 2018

BASE_DIR = '/home/lunet/gylc4/geodata/MIDAS/'
BASE_FILE = 'midas_rainhrly_{y}01-{y}12.txt'
STATIONS_DIR = os.path.join(BASE_DIR, 'stations')
STATIONS_LIST = os.path.join(BASE_DIR, 'src_id_list.xls')
OUT_FILE = '/home/lunet/gylc4/geodata/MIDAS/midas_precip_{s}-{e}.zarr'
SELECT_FILE = '../data/MIDAS/midas_{s}-{e}_precip_select.zarr'.format(s=START_YEAR, e=END_YEAR)
PAIR_FILE = '../data/MIDAS/midas_{s}-{e}_precip_pairs.nc'

# https://www.metoffice.gov.uk/public/weather/climate-extremes/#?tab=climateExtremes
UK_HOURLY_MAX = 92.0
UK_DAILY_MAX = 279.0


# http://artefacts.ceda.ac.uk/badc_datadocs/ukmo-midas/RH_Table.html
HEADER = ['end_time', 'id', 'id_type', 'ob_hour_count', 'version_num', 'met_domain_name',
          'src_id', 'rec_st_ind', 'prcp_amt', 'prcp_dur', 'prcp_amt_q', 'prcp_dur_q',
          'meto_stmp_time', 'midas_stmp_etime', 'prcp_amt_j'
          ]

IDX_COL_NAMES = ["id", "src_id", "src_name", "latitude", "longitude"]

DTYPES = {#'end_time':str,
          'id':np.uint32, 'id_type':str, 'ob_hour_count':np.uint16,
          'version_num':np.uint16, 'met_domain_name':str,
          'src_id':np.uint32, 'rec_st_ind':np.uint16,
          'prcp_amt':float, 'prcp_dur':float,
          'prcp_amt_q':str, 'prcp_dur_q':str,
          'meto_stmp_time':str, 'midas_stmp_etime':str, 'prcp_amt_j':str
          }

CHUNKS = {'end_time': -1, 'station': 30}
DTYPE = 'float32'
FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
INT_ENCODING = {'dtype': 'uint32', 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
STR_ENCODING = {'dtype': 'U', 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
GEN_ENCODING = {'compressor': zarr.Blosc(cname='lz4', clevel=9)}

KEEP_VARS = ['id', 'ob_hour_count', 'prcp_amt',
             'prcp_dur', 'rec_st_ind', 'src_id', 'version_num']

ENCODING = {'id': INT_ENCODING,
            # 'id_type': STR_ENCODING,
            'ob_hour_count': INT_ENCODING,
            'version_num': INT_ENCODING,
            # 'met_domain_name': STR_ENCODING,
            'src_id': INT_ENCODING,
            'rec_st_ind': INT_ENCODING,
            'prcp_amt': FLOAT_ENCODING,
            'prcp_dur': FLOAT_ENCODING,
            # 'prcp_amt_q': STR_ENCODING,
            # 'prcp_dur_q': STR_ENCODING,
            # 'meto_stmp_time': STR_ENCODING,
            # 'midas_stmp_etime': STR_ENCODING,
            # 'prcp_amt_j': STR_ENCODING
            }

def read_stations(data_dir, start_year, end_year):
    """Read station data from text file.
    Return a xarray DataArray.
    """
    prefix = 'midas_rainhrly_'
    suffix = '.txt'
    df_list = []
    for year in range(start_year, end_year+1):
        filename = BASE_FILE.format(y=year)
        print(filename)
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, sep=',', names=HEADER, na_values=' ',
                        dtype=DTYPES, engine='c',
                        index_col=False, parse_dates=['end_time'])
        df_list.append(df)

    # One dataframe per station
    df_all = pd.concat(df_list)
    gb = df_all.groupby('src_id')
    df_stations = [gb.get_group(x) for x in gb.groups]

    # Create a xarray DataArray
    df_stations_list = pd.read_excel(STATIONS_LIST, index_col=0)
    df_stations_list = df_stations_list[~df_stations_list.index.duplicated(keep='first')]
    # print(df_stations_list.head())
    ds_stations = []
    idx = [[] for i in range(len(IDX_COL_NAMES))]
    for df in df_stations:
        # delete duplicated index
        df.set_index('end_time', drop=True, inplace=True)
        df_dropped = df[~df.index.duplicated(keep='first')]
        ds = df_dropped.to_xarray()
        ds = ds.expand_dims('station')
        src_id = df_dropped['src_id'].unique()[0]  # station unique identifier
        ds.coords['station'] = [src_id]
        ds.drop('src_id')
        # Add other coordinates on the same dimension
        try:
            station_infos = df_stations_list.loc[[src_id]]
            src_name = station_infos['SRC_NAME'].values[0]
            lat = station_infos['HIGH_PRCN_LAT'].values[0]
            lon = station_infos['HIGH_PRCN_LON'].values[0]
        except KeyError:
            src_name = None
            lat = np.nan
            lon = np.nan
        ds.coords['src_name'] = xr.DataArray([src_name], dims='station')
        ds.coords['latitude'] = xr.DataArray([lat], dims='station')
        ds.coords['longitude'] = xr.DataArray([lon], dims='station')
        ds_stations.append(ds)

    ds_concat = xr.concat(ds_stations, dim='station')
    return ds_concat


def select_stations(ds, ymin, ymax):
    # 90% of all hourly records in 365 days
    min_records = int(365*24*.9)
    # Keep only the years of interest
    ds_short = ds.sel(end_time=slice(str(ymin), str(ymax)))
    # Keep only the quality controlled data
    # ds['qc'] = np.logical_or(ds_short['qc0'])
    ds_cc = ds_short.where(~ds_short['qc0'] & ~ds_short['qc1'] & ~ds_short['qc3'])
    # Number of values per year
    da_year_count = ds_cc['prcp_amt'].groupby('end_time.year').count(dim='end_time')
    min_years = int(len(da_year_count['year']) * .9)
    print(min_years)
    # number of full years per station
    num_of_full_years = da_year_count.where(da_year_count > min_records).count(dim='year')
    ds_cc['full_years'] = num_of_full_years
    # Stations with enough full years
    full_stations = num_of_full_years.where(num_of_full_years > min_years, drop=True)
    # print(full_stations.load())
    kept_station_code = full_stations['station'].values

    kept_col = KEEP_VARS + ['full_years']
    ds_sel = ds_cc.loc[{'station':kept_station_code}][kept_col]
    return ds_sel


def qc0(ds):
    """Flag data that are not hourly, and flagged as not reliable
    """
    ds['qc0'] = np.logical_or(np.not_equal(ds['version_num'], 1),
                              ds['ob_hour_count']!=1)



def qc1(ds):
    """Exceed UK hour record by 20%
    """
    ds['qc1'] = np.isfinite(ds['prcp_amt'].where(ds['prcp_amt'] >= UK_HOURLY_MAX * 1.2))
    # print(ds['qc1'].sum().load())
    # print(ds['prcp_amt'].min().load())



def qc3(ds):
    """Exceed UK 24h record by 20%
    """
    # Aggregate on 24h
    ds['prcp_daily'] = ds['prcp_amt'].rolling(end_time=24, min_periods=max(int(24*.9), 1)).sum(dim='end_time', skipna=True)
    ds['qc3'] = xr.where(ds['prcp_daily'] >= UK_DAILY_MAX * 1.2, True, False)


def qc4(ds):
    """Hourly totals at 0900 hours exceeding 2 times the mean daily rainfall for that month
    and preceded by 23h without rainfall
    """
    daily_mean_per_month = ds['prcp_daily'].groupby('end_time.month').mean(dim='end_time', skipna=True)
    print(daily_mean_per_month)
    prcp_0900 = ds['prcp_amt'].sel(end_time=datetime.time(9))
    month = prcp_0900['end_time.month']
    print(prcp_0900)
    over_thres = xr.where(prcp_0900 >= 2*daily_mean_per_month[month], True, False)
    print(over_thres.load())


def qc5(ds):
    """Three consecutive measurements at 0900 or 1200 hours following 23h without rainfall
    """
    pass


def qc7(ds):
    """Duplicate rainfall in consecutive hours that
    exceed two times the mean daily rainfall for that month.
    """
    pass


def quality_assessment(ds):
    qc0(ds)
    qc1(ds)
    qc3(ds)
    # qc4(ds)
    # print(ds['qc0'].sum().load())
    # print(ds['qc1'].sum().load())
    # print(ds['qc3'].sum().load())
    return ds


def to_gdf(ds):
    """return names as a geoDataframe
    """
    names = ds.src_name.load().drop('src_name')
    df = names.to_dataframe().dropna()
    df['geometry'] = [shapely.geometry.Point(lon, lat)
              for lon, lat in zip(df['longitude'], df['latitude'])]
    df['name'] = df['src_name'].str.decode("utf-8")
    del df['src_name']
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    return gdf


def main():
    with ProgressBar():
        # ds = read_stations(STATIONS_DIR, START_YEAR, END_YEAR).chunk(CHUNKS)
        # print(ds)
        # ds.to_zarr(OUT_FILE.format(s=START_YEAR, e=END_YEAR), mode='w')

        ds = xr.open_zarr(OUT_FILE.format(s=1950, e=2018))
        ds = quality_assessment(ds)
        ds_sel = select_stations(ds, START_YEAR, END_YEAR)#.chunk(CHUNKS)
        ds_loaded = ds_sel.load()
        print(ds_sel)
        # print(ds_sel['prcp_amt'])
        # print(ds_loaded['prcp_amt'].mean())
        # print(ds_loaded['prcp_amt'].min())
        # print(ds_loaded['prcp_amt'].max())
        # print(ds_loaded['ob_hour_count'].max())
        encoding = {'full_years': INT_ENCODING, **ENCODING}
        ds_sel.to_zarr(SELECT_FILE, mode='w', encoding=encoding)

        ds_sel = xr.open_zarr(SELECT_FILE)
        print(ds_sel)
        gdf = to_gdf(ds_sel)
        out_path = os.path.join('../data/MIDAS', "midas.gpkg")
        gdf.to_file(out_path, driver="GPKG")
        # print(ds_sel.max().load())

        # ds_pairs = station_pairs(ds_sel)
        # ds_pairs.load().to_netcdf(PAIR_FILE, mode='w')



if __name__ == "__main__":
    sys.exit(main())
