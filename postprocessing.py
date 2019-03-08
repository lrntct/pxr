# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime
import math
import subprocess

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry
import seaborn as sns
import cartopy as ctpy

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

import ev_quantiles
import helper


def convert_lon(longitude):
    """convert negative longitude into 360 longitude
    """
    return xr.where(longitude < 0, longitude + 360, longitude)
    # return np.where(longitude < 0, longitude + 360, longitude)

def get_site_list(ds, name_col, use_only=None):
    """return a dict {name: (lat, lon)}
    """
    # Set station coordinates to station name
    ds.coords['station'] = ds[name_col]

    if not use_only:
        use_only = ds[name_col].values

    sites = {}
    for name, lat, lon in zip(ds[name_col].values,
                              ds['latitude'].values,
                              ds['longitude'].values):
        if name in use_only:
            sites[name] = (lat, convert_lon(lon))
    return sites


def combine_ds_per_site(site_list, ds_cont=None, ds_points=None):
    """
    """
    if not ds_cont and not ds_cont:
        raise ValueError('No Dataset provided')

    keep_vars = ['gev', 'gev_scaling']

    # Extract sites values from the datasets. combine all ds along a new 'source' dimension
    ds_site_list = []
    for site_name, site_coord in list(site_list.items()):
        print(site_name)
        ds_site_sources = []
        # Select site on continous ds
        if ds_cont:
            for ds_name, ds in ds_cont.items():
                ds_cont_extract = (ds.sel(latitude=site_coord[0],
                                          longitude=convert_lon(site_coord[1]),
                                          method='nearest')
                                    # .drop(drop_coords)
                                    [keep_vars])
                ds_cont_extract = ds_cont_extract.expand_dims(['station','source'])
                ds_cont_extract.coords['station'] = [site_name]
                ds_cont_extract.coords['source'] = [ds_name]
                ds_site_sources.append(ds_cont_extract)
        # Add point ds
        if ds_points:
            for ds_name, (ds, name_col) in ds_points.items():
                ds.coords['station'] = ds[name_col]
                ds_pt_extract = (ds.sel(station=site_name)
                                 .drop(drop_coords + [name_col])
                                 [keep_vars])
                ds_pt_extract.coords['station'] = [site_name]
                ds_pt_extract.coords['source'] = [ds_name]
                ds_site_sources.append(ds_pt_extract)
        # Concatenate all sites along the source dimension
        ds_all_sources = xr.concat(ds_site_sources, dim='source')
        ds_site_list.append(ds_all_sources)

    # Concat all along the 'station' dimension
    ds_all = xr.concat(ds_site_list, dim='station')
    # Calculate regression lines
    sel_dict = dict(ev_param=['location', 'scale'])
    slope = ds_all['gev_scaling'].sel(scaling_param='slope', **sel_dict)
    intercept = ds_all['gev_scaling'].sel(scaling_param='intercept', **sel_dict)
    params_from_scaling = (10**intercept) * ds_all['duration']**slope
    # Add non-scaled shape
    da_shape = ds_all['gev'].sel(ev_param='shape').expand_dims(['ev_param'])
    da_shape.coords['ev_param'] = ['shape']
    ds_all['gev_scaled'] = xr.concat([params_from_scaling, da_shape], dim='ev_param').transpose(*ds_all['gev'].dims)
    return ds_all


def ds_to_df(ds, station_coord):
    # Create a dict of Pandas DF {'site': DF} with the df col prefixed with the source
    # print(ds)
    dict_df = {}
    for station in ds[station_coord].values:
        df_list = []
        for source in ds['source'].values:
            # get scale and location
            locator = {station_coord: station,
                       'source': source,
                       'ci': 'estimate',
                       'ev_param': ['location', 'scale']}
            ds_extract = ds.sel(**locator).drop(['gev_scaling', 'scaling_param'])
            # print(ds_extract)
            # expand the gev params to ds
            ds_gev_params = ds_extract['gev'].to_dataset(dim='ev_param')
            df = ds_gev_params.to_dataframe()
            rename_rules = {'location': '{}_location'.format(source),
                            'scale': '{}_scale'.format(source)}
            df.rename(columns=rename_rules, inplace=True)
            # print(df.head())
            df_list.append(df)
            # Get values from regression
            ds_gev_params = ds_extract['gev_scaled'].to_dataset(dim='ev_param')
            df2 = ds_gev_params.to_dataframe()
            rename_rules = {
                'location': '{}_location_lr'.format(source),
                'scale': '{}_scale_lr'.format(source)
                }
            df2.rename(columns=rename_rules, inplace=True)
            df_list.append(df2)
            # print(df)
        df_all = pd.concat(df_list, axis=1, sort=False)
        # print(df_all.head())
        dict_df[station] =  df_all
    return dict_df


def get_quantile_dict(quantiles, **kwargs):
    """
    kwargs: a dict of 'source': (ds, dim)
    return df_dict: {param: [(source, df), ]}
    """
    df_dict = {}
    for param in ['location', 'scale']:
        # Actual values of the regression lines
        df_list = []
        for source, (ds, dim) in kwargs.items():
            try:
                ds_daily = ds.sel(scaling_extent=b'daily')
            except KeyError:
                ds_daily = ds.sel(scaling_extent='daily')
            slope = ds_daily['{}_line_slope'.format(param)]
            intercept = ds_daily['{}_line_intercept'.format(param)]
            log_reg = np.log10(10**intercept * ds_daily['duration']**slope)
            diff = log_reg - ds_daily['log_{}'.format(param)]
            diff_q = diff.compute().quantile(quantiles, dim=dim)
            df_q = diff_q.to_dataset('quantile').to_dataframe()
            df_list.append((source, df_q))
        df_dict[param] = df_list
    return df_dict


def prepare_midas_mean(ds_era, ds_midas, ds_midas_mean):
    keep_vars = ['location', 'scale', 'log_location', 'log_scale',
                 'location_line_slope', 'scale_line_slope',
                 'location_line_intercept', 'scale_line_intercept']
    # convert station names to UTF8
    names_str = np.array([b.decode('utf8') for b in ds_midas['src_name'].values])
    ds_midas['src_name'].values = names_str
    # Set station coordinates to station name
    ds_midas.coords['station'] = ds_midas['src_name']
    ds_midas = ds_midas.drop('src_name')

    ds_list = []
    for pair_name in ds_midas_mean['pair']:
        pair_name = str(pair_name.values)
        ds_pair_list = []
        # Find MIDAS stations in pairs
        for i, s in enumerate(pair_name.split('--')):
            site_num = i+1
            ds_s = ds_midas[keep_vars].sel(station=s, drop=True)
            site_lat = ds_s['latitude'].values
            site_lon = convert_lon(ds_s['longitude'].values)
            # Add a source coordinate
            ds_s = ds_s.expand_dims(['source', 'pair'])
            ds_s.coords['source'] = ['MIDAS s'+str(site_num)]
            ds_s.coords['pair'] = [pair_name]
            ds_pair_list.append(ds_s.drop(['latitude', 'longitude']))

        # add the source dim to the other two datasets
        ds_mean = ds_midas_mean[keep_vars].sel(pair=pair_name)
        ds_mean = ds_mean.expand_dims(['pair', 'source'])
        ds_mean.coords['source'] = ['MIDAS mean']
        ds_pair_list.append(ds_mean)

        ds_era_extract = ds_era[keep_vars].sel(latitude=site_lat,
                                               longitude=site_lon,
                                               method='nearest', drop=True)
        ds_era_extract.coords['scaling_extent'] = ds_mean['scaling_extent']
        ds_era_extract = ds_era_extract.expand_dims(['pair', 'source'])
        ds_era_extract.coords['pair'] = [pair_name]
        ds_era_extract.coords['source'] = ['ERA5']
        ds_pair_list.append(ds_era_extract)

        # Join all ds of the cell in a single dataset
        ds_site = xr.concat(ds_pair_list, dim='source')
        ds_list.append(ds_site)

    # Join all along the pair coordinate
    ds_all = xr.concat(ds_list, dim='pair')
    # Calculate regression lines
    for p in ['location', 'scale']:
        dur = ds_all['duration']
        slope = ds_all['{}_line_slope'.format(p)]
        intercept = ds_all['{}_line_intercept'.format(p)]
        linereg_var = '{}_lr'.format(p)
        ds_all[linereg_var] = (10**intercept) * dur**slope
    return ds_all


def estimate_intensities(ds_era, ds_gauges):
    """
    """
    # keep only stations with coordinates
    ds_gauges = ds_gauges.reset_coords(['latitude', 'longitude'])
    gauges_sel = np.logical_and(np.isfinite(ds_gauges['latitude']), np.isfinite(ds_gauges['longitude']))
    ds_gauges = ds_gauges.where(gauges_sel, drop=True)
    # Drop Gibraltar
    ds_gauges = ds_gauges.drop([1585], dim='station')
    # Convert gauges longitudes
    ds_gauges['longitude'] = convert_lon(ds_gauges['longitude'])
    # Select the cells above the stations
    ds_era_sel = ds_era.sel(latitude=ds_gauges['latitude'],
                            longitude=ds_gauges['longitude'],
                            method='nearest')
    ds_list = []
    # print(ds_era_sel['longitude'])
    # print(ds_gauges['longitude'])
    for ds, name in zip([ds_era_sel, ds_gauges], ['ERA5', 'MIDAS']):
        # Delete coordinates (no longer used)
        ds = ds.drop(['latitude', 'longitude', 'src_name'])
        # Merge the two datasets along a new dimension
        ds = ds.expand_dims('source')
        ds.coords['source'] = [name]
        ds_list.append(ds)
    ds = xr.merge(ds_list)

    # GEV param from scaling
    sel_dict = dict(source='ERA5', ev_param=['location', 'scale'])
    slope = ds['gev_scaling'].sel(scaling_param='slope', **sel_dict)
    intercept = ds['gev_scaling'].sel(scaling_param='intercept', **sel_dict)
    params_from_scaling = (10**intercept) * ds['duration']**slope
    params_from_scaling = params_from_scaling.expand_dims('source')#.drop('scaling_param')
    params_from_scaling.coords['source'] = ['ERA5_scaled']
    # Add non-scaled shape.
    # print(params_from_scaling)
    da_shape =  ds['gev'].sel(source='ERA5', ev_param='shape')
    da_shape = da_shape.expand_dims(['source', 'ev_param'])
    da_shape.coords['source'] = ['ERA5_scaled']
    da_shape.coords['ev_param'] = ['shape']
    # print(da_shape)
    params_from_scaling = xr.concat([params_from_scaling, da_shape], dim='ev_param')
    # print(params_from_scaling)
    gev_params = xr.concat([params_from_scaling, ds['gev']], dim='source')
    # print(gev_params)

    # calculate intensity for given duration and return period
    da_list = []
    for T in [2,10,50,100,500,1000]:
        loc = gev_params.sel(ev_param='location', ci='estimate')
        scale = gev_params.sel(ev_param='scale', ci='estimate')
        shape = gev_params.sel(ev_param='shape', ci='estimate')
        intensity = ev_quantiles.gev_quantile(T, loc, scale, shape).rename('intensity')
        intensity = intensity.expand_dims('T')
        intensity.coords['T'] = [T]
        da_list.append(intensity)
    da_i = xr.concat(da_list, dim='T').drop(['ci'])
    ds_i = da_i.to_dataset()

    # Robust regression and MAE
    new_sources = []
    for reg_source in ['ERA5', 'ERA5_scaled']:
        i_midas = da_i.sel(source='MIDAS')
        i_source = da_i.sel(source=reg_source)
        slope = helper.RLM_slope(i_midas, i_source, dim='station')
        slope = slope.rename('arf').expand_dims('source')
        slope.coords['source'] = [reg_source]

        reg_line = slope * i_midas
        rlm_name = reg_source + '_rlm'
        reg_line = reg_line.rename('intensity').transpose(*da_i.dims)
        reg_line.coords['source'] = [rlm_name]
        # MAE
        mae = np.abs(i_source - i_midas).mean(dim='station').rename('mae')
        mae = mae.expand_dims('source')
        mae.coords['source'] = [reg_source]
        ds = xr.merge([slope, reg_line, mae])
        new_sources.append(ds)
    ds = xr.auto_combine([ds_i] + new_sources, concat_dim='source')
    return ds


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
