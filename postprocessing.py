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

import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
from uncertainties import ufloat, unumpy


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

    keep_vars = ['annual_max', 'gev', 'gev_scaled', 'gev_scaling']

    # Extract sites values from the datasets. combine all ds along a new 'source' dimension
    ds_site_list = []
    for site_name, site_coord in list(site_list.items()):
        # print(site_name)
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
    return ds_all


def ds_to_df(ds, station_coord):
    # Create a dict of Pandas DF {'site': DF} with the df col prefixed with the source
    # print(ds)
    ci_l = '0.025'
    ci_h = '0.975'
    dict_df = {}
    for station in ds[station_coord].values:
        df_list = []
        for source in ds['source'].values:
            locator = {station_coord: station,
                       'source': source,
                       'ci': ['estimate', ci_l, ci_h],  # 95% CI
                       'ev_param': ['location', 'scale']}
            ds_extract = ds.sel(**locator).drop(['gev_scaling', 'scaling_param'])

            ds_gev_params = ds_extract['gev'].to_dataset(dim='ev_param')
            ds_gev_params_scaled = ds_extract['gev_scaled'].to_dataset(dim='ev_param')
            df_param_list = []
            for param_name in ['location', 'scale']:
                # get scale and location
                drop_coors = ['source', 'latitude', 'longitude', 'station']
                ds_gev_param = ds_gev_params[param_name].to_dataset(dim='ci').drop(drop_coors)
                df_param = ds_gev_param.to_dataframe()
                rename_rules = {'estimate': '{}_{}_est'.format(source, param_name),
                                ci_l: '{}_{}_ci_l'.format(source, param_name),
                                ci_h: '{}_{}_ci_h'.format(source, param_name),
                                }
                df_param.rename(columns=rename_rules, inplace=True)
                # if station == 'Jakarta':
                #     print(df_param.head())
                df_param_list.append(df_param)
                # Get params from regression
                ds_gev_param = ds_gev_params_scaled[param_name].to_dataset(dim='ci').drop(drop_coors)
                df_param = ds_gev_param.to_dataframe()
                rename_rules = {'estimate': '{}_{}_lr_est'.format(source, param_name),
                                ci_l: '{}_{}_lr_ci_l'.format(source, param_name),
                                ci_h: '{}_{}_lr_ci_h'.format(source, param_name),
                                }
                df_param.rename(columns=rename_rules, inplace=True)
                # if station == 'Jakarta':
                #     print(df_param.head())
                df_param_list.append(df_param)
            df = pd.concat(df_param_list, axis=1, sort=False)
            # print(df.head())
            df_list.append(df)
        df_all = pd.concat(df_list, axis=1, sort=False)
        # print(df_all.head())
        dict_df[station] =  df_all
    return dict_df


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


def estimate_intensity(gev_params, da_n, T):
    """Estimate intensity for return period T and its CI.
    gev_params: a DataArray of GEV parameters
    """
    # Approximate the number of observations
    n_obs = da_n.min().values.item()
    # Get intensities
    loc = gev_params.sel(ev_param='location', ci='estimate')
    scale = gev_params.sel(ev_param='scale', ci='estimate')
    shape = gev_params.sel(ev_param='shape', ci='estimate')
    i_estimate = ev_quantiles.gev_quantile(T, loc, scale, shape).rename('intensity')
    # Confidence interval
    var_i = ev_quantiles.gev_quantile_var_fixed_shape(T, c1=1.1412, c2=0.8216, c3=1.2546,
                                         da_gev=gev_params, n_obs=n_obs)
    ci_i_list = []
    for ci in gev_params['ci']:
        ci = ci.item()
        if ci in ['estimate', '0.500']:
            i_ci = i_estimate
        else:
            t_quantile = scipy.stats.t.ppf(float(ci), n_obs-2)
            i_ci = i_estimate + t_quantile * np.sqrt(var_i/n_obs)
        i_ci = i_ci.expand_dims('ci')
        i_ci.coords['ci'] = [ci]
        ci_i_list.append(i_ci)
    intensity = xr.concat(ci_i_list, dim='ci')
    return intensity


def estimate_intensities_errors(ds_era, ds_gauges):
    """
    """
    # keep only stations with coordinates
    ds_gauges = ds_gauges.reset_coords(['latitude', 'longitude'])
    gauges_sel = np.logical_and(np.isfinite(ds_gauges['latitude']), np.isfinite(ds_gauges['longitude']))
    ds_gauges = ds_gauges.where(gauges_sel, drop=True)
    # Convert gauges longitudes
    ds_gauges['longitude'] = convert_lon(ds_gauges['longitude'])
    # Select the cells above the stations
    ds_era_sel = ds_era.sel(latitude=ds_gauges['latitude'],
                            longitude=ds_gauges['longitude'],
                            method='nearest')
    ds_list = []
    for ds, name in zip([ds_era_sel, ds_gauges], ['ERA5', 'MIDAS']):
        # Delete coordinates (no longer used)
        ds = ds.drop(['latitude', 'longitude', 'src_name'])
        # Merge the two datasets along a new dimension
        ds = ds.expand_dims('source')
        ds.coords['source'] = [name]
        ds_list.append(ds)
    ds = xr.merge(ds_list)

    # GEV param from scaling
    gev_scaled = ds['gev_scaled'].sel(source='ERA5').expand_dims('source')
    gev_scaled.coords['source'] = ['ERA5_scaled']
    gev_params = xr.concat([gev_scaled, ds['gev']], dim='source')

    # calculate intensity for given duration and return period
    da_list = []
    for T in [2,10,50,100,500,1000]:
        intensity = estimate_intensity(gev_params, ds['n_obs'], T)
        intensity = intensity.expand_dims('T')
        intensity.coords['T'] = [T]
        da_list.append(intensity)
    da_i = xr.concat(da_list, dim='T').sel(ci='estimate').drop('ci')
    ds_i = da_i.to_dataset()
    # print(ds_i)

    # Robust regression and error
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
        # MAPE
        mape = np.abs((i_source - i_midas) / i_midas).mean(dim='station')
        mape = mape.expand_dims('source').rename('mape')
        mape.coords['source'] = [reg_source]
        # MPE + ci --- Change type to propagate uncertainty, 95% CI
        # ui_source = unumpy.uarray(i_source.sel(ci='estimate'),
        #                           i_source.sel(ci='0.975')-i_source.sel(ci='estimate'))
        # ui_midas = unumpy.uarray(i_midas.sel(ci='estimate'),
        #                           i_midas.sel(ci='0.975')-i_midas.sel(ci='estimate'))
        # pe = (ui_source - ui_midas) / ui_midas
        # station_ax_num = i_source.sel(ci='estimate').get_axis_num('station')
        # mpe = pe.mean(axis=station_ax_num)
        # mpe_nom = unumpy.nominal_values(mpe)
        # mpe_ci = unumpy.std_devs(mpe)

        # da_orig_sel = i_source.sel(ci='estimate').isel(station=0)
        # mpe = xr.DataArray(mpe_nom, coords=da_orig_sel.coords,
        #                    dims=da_orig_sel.dims,
        #                    name='mpe')
        # pe_ci_l = mpe - mpe_ci
        # pe_ci_h = mpe + mpe_ci
        pe = (i_source - i_midas) / i_midas
        pe_q = pe.compute().quantile([0.025, 0.5, 0.975], dim='station')
        pe_q = pe_q.expand_dims('source')
        mdpe = pe_q.sel(quantile=0.5, drop=True)
        mdpe.coords['source'] = [reg_source]
        pe_ci_l = pe_q.sel(quantile=0.025, drop=True)
        pe_ci_h = pe_q.sel(quantile=0.975, drop=True)
        pe_ci_l.coords['source'] = [reg_source + '_ci_l']
        pe_ci_h.coords['source'] = [reg_source + '_ci_h']
        mpe = xr.concat([mdpe, pe_ci_h, pe_ci_l], dim='source').rename('mdpe')
        print(mpe)
        # Merge in a DS
        ds = xr.merge([slope, reg_line, mae, mape, mpe])
        new_sources.append(ds)
    ds = xr.auto_combine([ds_i] + new_sources, concat_dim='source')
    return ds


def adequacy(da_mape, threshold=.2):
    for source in ['ERA5', 'ERA5_scaled']:
        print(source)
        da_sel = da_mape.sel(source=source).load()
        da_adequate = da_sel <= threshold
        adq = da_adequate.where(da_adequate, drop=True)
        print(adq)


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
            da_gev = ds['gev'].sel(ci='estimate', ev_param=param)
            da_scaled = ds['gev_scaled'].sel(ci='estimate', ev_param=param)
            # diff = np.log10(da_scaled / da_gev)
            diff = (da_scaled - da_gev) / da_gev
            diff_q = diff.compute().quantile(quantiles, dim=dim)
            df_q = diff_q.to_dataset('quantile').to_dataframe()
            df_list.append((source, df_q))
        df_dict[param] = df_list
    return df_dict


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
