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


# Headless plotting
import matplotlib
matplotlib.use("Agg")

import sys
import os
import math
import subprocess

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry
import seaborn as sns
import toml
import cartopy as ctpy
import scipy.stats

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

import config
import ev_quantiles
import helper
import postprocessing

# Increase DPI of figures
matplotlib.rcParams['savefig.dpi'] = 200

plot_dir = config.plot['dir']

midas_stations = config.plot['station_map']
basemap = config.plot['base_map']

EXTRACT = dict(latitude=slice(45, -45),
               longitude=slice(0, 180))

# Coordinates of study sites
# STUDY_SITES = {'Kampala': (0.317, 32.616),
#                'Kisumu': (0.1, 34.75)}
STUDY_SITES = {
               'Jakarta': (-6.2, 106.816), 'Sydney': (-33.865, 151.209),
               'Beijing': (39.92, 116.38), 'New Delhi': (28.614, 77.21),
               'Niamey': (13.512, 2.125), #'Jeddah': (21.54, 39.173), 'Cape Town': (-33.925278, 18.4238),
               'Nairobi': (-1.28, 36.82), #'Brussels': (50.85, 4.35),
               'Santiago': (-33.45, -70.67), #'New York City': (40.72, -74.0),
               'Mexico City': (19.43, -99.13), # 'Vancouver': (49.25, -123.1),
            #    'Vienna': (48.2, 16.367),
               'Natal': (-5.78, -35.2),
            #    'Surakarta': (-7.57, 110.817), 'Semarang': (-6.97, 110.417)
               }

XTICKS = [1, 3, 6, 12, 24, 48, 120, 360]

# color-blind safe qualitative palette from colorbrewer2 (paired)
C_PRIMARY_1 = '#1f78b4'
C_SECONDARY_1 = '#6ba9ca'
C_PRIMARY_2 = '#33a02c'
C_SECONDARY_2 = '#89c653'

METADATA = 'metadata.toml'

with open(METADATA, 'r') as toml_file:
    metadata = toml.loads(toml_file.read())

# output
PXR2 = f"pxr2-{metadata['common']['version']}.nc"
PXR4 = f"pxr4-{metadata['common']['version']}.nc"


def set_logd_xticks(ax, xmin=min(XTICKS), xmax=max(XTICKS)):
    xticks = [i for i in XTICKS if i <= xmax and i >= xmin]
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


def single_map(da, cbar_label, title, fig_name, center=False, reverse=False):
    """https://cbrownley.wordpress.com/2018/05/15/visualizing-global-land-temperatures-in-python-with-scrapy-xarray-and-cartopy/
    """
    plt.figure(figsize=(8, 5))
    ax_p = plt.gca(projection=ctpy.crs.EqualEarth(), aspect='auto')
    if center is True:
        if reverse:
            cmap = 'RdBu_r'
        else:
            cmap = 'RdBu'
    else:
        cmap='viridis'
    da.plot.imshow(ax=ax_p, transform=ctpy.crs.PlateCarree(),
                   robust=True, cmap=cmap, center=center,
                   cbar_kwargs=dict(orientation='horizontal', label=cbar_label))
    ax_p.coastlines(linewidth=.5, color='black')
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def multi_maps(da_list, disp_names, value_name, fig_name,
               sqr=False, center=False, reverse=False, robust=True,
               onebar=True, bar_labels=None):
    crs = ctpy.crs.EqualEarth()
    # reshape variables as dimension
    da_list2 = []
    for da in da_list:
        if sqr == True:
            da_sel = np.square(da).expand_dims('param')
        else:
            da_sel = da.expand_dims('param')
        da_sel.coords['param'] = [da.name]
        da_list2.append(da_sel)
    da_all = xr.concat(da_list2, 'param')
    da_all.attrs['long_name'] = value_name  # Color bar title
    # Actual plot
    # print(da_all['longitude'])
    aspect = len(da_all['longitude']) / len(da_all['latitude'])
    if center is True:
        if reverse:
            cmap = 'RdBu_r'
        else:
            cmap = 'RdBu'
    else:
        cmap = 'viridis'
    if onebar:  # One shared colorbar
        fg = da_all.plot(col='param', col_wrap=1,
                         transform=ctpy.crs.PlateCarree(), aspect=aspect,
                         cmap=cmap, center=center,
                         robust=robust, extend='both',
                         subplot_kws=dict(projection=crs)
                         )
        for ax, disp_name in zip(fg.axes.flat, disp_names):
            ax.coastlines(linewidth=.5, color='black')
            ax.set_title(disp_name)
    else:
        fig, axes = plt.subplots(len(da_list2), 1, sharey=False, sharex=False,
                                subplot_kw=dict(projection=crs),
                                 )
        for da, ax, disp_name, bar_label in zip(da_list2, axes, disp_names, bar_labels):
            da = da.isel(param=0, drop=True)
            # print(da)
            da.plot.imshow(ax=ax, transform=ctpy.crs.PlateCarree(),
                           robust=True, cmap=cmap, center=center,
                           cbar_kwargs=dict(orientation='vertical', label=bar_label))
            ax.coastlines(linewidth=.5, color='black')
            ax.set_title(disp_name)
        plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


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
            sites[name] = (lat, postprocessing.convert_lon(lon))
    return sites


def plot_point_map(ds, ax):
    ds_sel = ds.sel(duration=ds.duration.values[0],
                    source=ds.source.values[0],)
    df = ds_sel.to_dataframe()
    df['geometry'] = [shapely.geometry.Point(lon, lat)
                      for lon, lat in zip(df['longitude'],
                                          df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    ax.set_global()
    ax.coastlines(linewidth=.3, color='0.6', zorder=0)
    gdf.plot(ax=ax, markersize=25, transform=ctpy.crs.PlateCarree(), zorder=20)
    ax.set_title('Sample sites location')


def plot_scaling_per_site(ds, fig_name):
    """
    """
    linesyles = {
        # 'MIDAS_location': dict(linestyle='None', linewidth=0, marker='v', markersize=3,
        #                        color=C_PRIMARY_1, label='Location $\mu$ (MIDAS)'),
        # 'MIDAS_location_lr': dict(linestyle='solid', linewidth=0.5, marker=None, markersize=0,
        #                           color=C_PRIMARY_1, label='$a d^\\alpha$ (MIDAS)'),
        # 'MIDAS_scale': dict(linestyle='None', linewidth=0, marker='v', markersize=3,
        #                     color=C_PRIMARY_2, label='Scale $\sigma$ (MIDAS)'),
        # 'MIDAS_scale_lr': dict(linestyle='solid', linewidth=0.5, marker=None, markersize=0,
        #                        color=C_PRIMARY_2, label='$\sigma_D (d/D)^\\beta$ (MIDAS)'),
        'ERA5_location': dict(linestyle='None', linewidth=0, marker='o', markersize=2,
                              color=C_PRIMARY_1, label='Location $\mu$', zorder=20),
        'ERA5_scale': dict(linestyle='None', linewidth=0, marker='o', markersize=2,
                           color=C_PRIMARY_2, label='Scale $\sigma$', zorder=20),
        'ERA5_location_lr': dict(linestyle='solid', linewidth=1., marker=None, markersize=0,
                                 color=C_PRIMARY_1, label='$ad^\\alpha$', zorder=10),
        'ERA5_scale_lr': dict(linestyle='solid', linewidth=1., marker=None, markersize=0,
                              color=C_PRIMARY_2, label='$bd^\\beta$', zorder=10),
        # 'ERA5_location_lr_daily': dict(linestyle='dashed', linewidth=1., marker=None, markersize=0,
        #                          color=C_SECONDARY_1, label='$ad^\\alpha$ (daily)'),
        # 'ERA5_scale_lr_daily': dict(linestyle='dashed', linewidth=1., marker=None, markersize=0,
        #                       color=C_SECONDARY_2, label='$bd^\\beta$ (daily)'),
                }

    dict_df = postprocessing.ds_to_df(ds, 'station')
    # sys.exit()  ###
    col_num = 2
    row_num = math.ceil(len(dict_df) / col_num)
    # fig_size = (9, 5)  # landscape
    fig_size = (6, 9)  # portrait
    fig = plt.figure(figsize=fig_size)
    ax_num = 1

    # Draw map
    ax_map = fig.add_subplot(row_num, col_num, ax_num,
                             projection=ctpy.crs.EqualEarth(),
                             aspect='auto')
    plot_point_map(ds, ax_map)
    ax_num += 1

    sites_ax_list = []
    for site_ax_num, (site_name, df) in enumerate(dict_df.items()):
        if site_ax_num == 0:
            ax = fig.add_subplot(row_num, col_num, ax_num)
            first_ax = ax
        else:
            ax = fig.add_subplot(row_num, col_num, ax_num,
                                 sharex=first_ax, sharey=first_ax)
        print(site_name)
        # if site_name == 'Jakarta':
        #     print(df[['ERA5_location_ci_l', 'ERA5_location_ci_h', 'ERA5_location_lr_ci_l', 'ERA5_location_lr_ci_h']].head())
        # plot
        for col_prefix, styles in linesyles.items():
            col_est = col_prefix + '_est'
            col_ci_l = col_prefix + '_ci_l'
            col_ci_h = col_prefix + '_ci_h'
            # plot estimate
            df[col_est].plot(ax=ax, label=styles['label'],
                title=site_name, zorder=styles['zorder'],
                loglog=True,
                linestyle=styles['linestyle'], linewidth=styles['linewidth'],
                markersize=styles['markersize'],
                marker=styles['marker'], color=styles['color'])
            # plot error
            if col_prefix.endswith('_lr'):
                ax.fill_between(df.index, df[col_ci_h], df[col_ci_l],
                        alpha=0.25, label=styles['label'] + ' 95% CI', linewidth=0,
                        color=styles['color'], zorder=0)
            else:
                df[col_ci_h].plot(ax=ax, label=styles['label'] + ' 95% CI',
                    title=site_name, loglog=True,
                    zorder=5, alpha=0.9,  marker='_',
                    linestyle=styles['linestyle'], linewidth=styles['linewidth'],
                    markersize=styles['markersize'],
                    color=styles['color'])
                df[col_ci_l].plot(ax=ax, label='',
                    title=site_name, loglog=True,
                    zorder=5, alpha=0.9, marker='_',
                    linestyle=styles['linestyle'], linewidth=styles['linewidth'],
                    markersize=styles['markersize'],
                    color=styles['color'])
        lines, labels = ax.get_legend_handles_labels()
        ax.set_xlabel('$d$ (hours)')
        ax.set_ylabel('$\mu, \sigma$ (mm/h)')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        sites_ax_list.append(ax)
        ax_num += 1

    # set ticks
    for ax in sites_ax_list:
        set_logd_xticks(ax)
        ax.use_sticky_edges = False
        ax.autoscale()
    # lgd_ncol = math.ceil(len(labels) / 4)  # talk (landscape)
    lgd_ncol = math.ceil(len(labels) / 2)  # article (portrait)
    lgd = fig.legend(lines, labels, loc='lower center', ncol=lgd_ncol)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=.2, wspace=None, hspace=None)  # Talk (landscape)
    plt.subplots_adjust(bottom=.12, wspace=None, hspace=None)  # Article (portrait)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_hourly(ds, sites, fig_name):
    ds_list = []
    for site_name, site_coord in sites.items():
        ds_list.append(ds.sel(latitude=site_coord[0],
                       longitude=site_coord[1],
                       method='nearest'))
    drop_coords = ['latitude', 'longitude']
    ds_sites = xr.concat(ds_list, dim='site').drop(drop_coords)
    ds_sites.coords['site'] = list(sites.keys())
    p = ds_sites['precipitation'].plot(col='site')
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def probability_plot(ds, sites, fig_name):
    # print(ds)
    DURATION = 6
    keep_var = ['ecdf_gringorten', 'annual_max', 'cdf', 'ev_parameter']
    # extract sites values from the dataset
    dict_cdf = {}
    ds_list = []
    for site_name, site_coord in sites.items():
        ds_sel = ds.sel(latitude=site_coord[0],
                        longitude=site_coord[1],
                        duration=DURATION,
                        method='nearest', drop=True)
        ds_sel = ds_sel.expand_dims('site')
        ds_sel.coords['site'] = [site_name]
        ds_list.append(ds_sel[keep_var])
    # Split CDF into variables
    ds_sites = xr.concat(ds_list, dim='site')
    ds_cdf = ds_sites['cdf'].to_dataset(dim='ev_fit')
    ds_split = xr.merge([ds_sites.drop('cdf'), ds_cdf]).drop('ev_fit')
    # All the probabilities into one dimension
    da_prob = ds_split.drop('annual_max').to_array('cdf', name='prob')
    ds_prob = xr.merge([da_prob, ds_split['annual_max']])
    # print(ds_prob)

    # compute quantiles for a range of years
    ds_params = ds_sites['ev_parameter'].to_dataset(dim='ev_param')
    # print(ds_params)
    t = np.arange(2, 10000)
    return_periods = xr.DataArray(t, dims=['T'], coords={'T': t})
    # print(return_periods)
    da_q = ev_fit.gev_quantile(return_periods,
                                       ds_params['location'],
                                       ds_params['scale'],
                                       ds_params['shape']
                              ).rename('intensity')
    # print(da_q)
    # to Pandas dataframe
    df_cdf = (ds_prob
              .to_dataframe()
              .reset_index()
              .drop('year', axis='columns')
            )
    # prob to return period
    df_cdf['T'] = 1/(1-df_cdf['prob'])
    df_cdf = df_cdf.sort_values('T')

    # plot
    colwrap = 3
    row_num = math.ceil(len(sites) / colwrap)
    fig, axes = plt.subplots(row_num, colwrap, sharey=False, sharex=True, figsize=(9, 7))
    for ax, site_name in zip(axes.flat, sites.keys()):
        df_site = df_cdf.loc[df_cdf['site'] == site_name]
        # ECDF
        ecdf = df_site.loc[df_cdf['cdf'].str.startswith('ecdf_')]
        ecdf.plot(x='T', y='annual_max', logx=True, legend=False, ax=ax,
                  linestyle='', lw=0.1, marker='x', markeredgecolor='.1',
                  markersize=5, label='ecdf')
        # CDFs
        gumbel = da_q.sel(site=site_name, ev_fit='gumbel').to_dataframe().reset_index()
        gumbel.plot(x='T', y='intensity', logx=True, legend=False, ax=ax,
                    c='.2', linewidth=1, linestyle='--', label='Gumbel')
        gev = da_q.sel(site=site_name, ev_fit='gev').to_dataframe().reset_index()
        gev.plot(x='T', y='intensity', logx=True, legend=False, ax=ax,
                       label='GEV')
        frechet = da_q.sel(site=site_name, ev_fit='frechet').to_dataframe().reset_index()
        frechet.plot(x='T', y='intensity', logx=True, legend=False, ax=ax,
                       label='Fréchet')

        ax.set_title(site_name)
        ax.set_ylabel('i (mm/h)')
        ax.set_xlabel('return period (years)')
        lines, labels = ax.get_legend_handles_labels()

    lgd_labels = ['Observed values', 'Gumbel', 'GEV', 'Fréchet']
    lgd = fig.legend(lines, lgd_labels, loc='lower center', ncol=4)

    # plt.tight_layout()
    plt.subplots_adjust(bottom=.24, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_gauges_data(ds, ymin, ymax, fig_name):
    """Plot the completeness of the gauges data
    """
    da_year_count = ds['precipitation'].groupby('date.year').count(dim='date')
    year_count_sel = da_year_count.sel(year=slice(ymin, ymax))
    # plot
    p = year_count_sel.plot(vmin=int(365*.9))
    ax = plt.gca()
    labels = list(da_year_count['code'].values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_gauges_map_from_ds(ds, id_dim, fig_name, global_extent=True):
    """convert dataset in geopandas DataFrame
    plot it
    """
    # Create a GeoDataFrame
    ds_sel = ds.sel(duration=24, year=2000)#['annual_max']
    df = ds_sel.to_dataframe().set_index(id_dim, drop=True).drop(axis=1, labels=['annual_max', 'year', 'duration'])
    df['geometry'] = [shapely.geometry.Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(
        df, geometry='geometry')#.cx[slice(-2, -1.5), slice(51, 52)]
    print(gdf)

    # Plot the gauges on a world map
    plt.figure(figsize=(12, 12))
    if global_extent:
        ax_p = plt.gca(projection=ctpy.crs.EqualEarth(), aspect='auto')
        ax_p.set_global()
    else:
        ax_p = plt.gca(projection=ctpy.crs.PlateCarree(), aspect='auto')
        gl = ax_p.gridlines(crs=ctpy.crs.PlateCarree(), linewidth=.3,
                            #color='black', alpha=0.5, linestyle='--',
                            draw_labels=False,
                            xlocs=np.arange(-10,10,0.25),
                            ylocs=np.arange(45,70,0.25),
                            )
    ax_p.coastlines(linewidth=.3, color='black')
    gdf.plot(ax=ax_p, markersize=5, transform=ctpy.crs.PlateCarree())
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def fig_gauges_map(fig_name):
    source_crs = ctpy.crs.PlateCarree()
    display_crs =  ctpy.crs.OSGB()
    midas = gpd.read_file(midas_stations)
    ne_land = gpd.read_file(basemap)

    # Extent of the UK
    xmin = -7
    xmax = 2
    ymin = 49.5
    ymax = 59.5
    aspect = (xmax-xmin) / (ymax-ymin)
    fig_width = 5
    fig = plt.figure(figsize=(fig_width, fig_width*aspect))
    ax_p = fig.gca(projection=display_crs,
                   aspect=aspect)
    ticks = [0,1,2]

    ne_land.plot(ax=ax_p, color='', edgecolor='0.2',
                 linewidth=0.4, alpha=0.4, transform=source_crs)
    midas.plot(ax=ax_p, color='k', marker='x', markersize=15,
               linewidth=1.5, label='MIDAS station', transform=source_crs)

    ax_p.set_extent([xmin, xmax, ymin, ymax], crs=source_crs)
    gl = ax_p.gridlines(crs=source_crs, linewidth=.3,
                        alpha=0.4,
                        )

    fig_path = os.path.join(plot_dir, fig_name)
    plt.savefig(fig_path)
    subprocess.call(['pdfcrop', '--margins', '10', fig_path, fig_path])
    plt.close()


def hexbin(da1, da2, xlabel, ylabel, fig_name):
    x = da1.values.flat
    y = da2.values.flat
    xq = np.nanquantile(x, [0.01, 0.99])
    yq = np.nanquantile(y, [0.01, 0.99])
    xmin, xmax = xq[0], xq[1]
    ymin, ymax = xq[0], xq[1]

    fig, ax = plt.subplots(figsize=(4, 3))
    # ax.axis([xmin, xmax, ymin, ymax])
    hb = ax.hexbin(x, y, gridsize=20, extent=[xmin, xmax, ymin, ymax],
                   mincnt=100, bins=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot([xmin, xmax], [ymin, ymax], color='k', linewidth=0.5)
    norm = matplotlib.colors.Normalize(vmin=1000, vmax=10000)
    cb = plt.colorbar(hb, spacing='uniform', orientation='vertical',
                      label='Number of cells', norm=norm, extend='both')
    # cb.set_label('# of cells')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def fig_map_KS(ds):
    """create maps of the Kolmogorov-Smirnov / Lilliefors test statistic
    """
    alpha = 0.05
    DURATION = 2
    Dcrit = ds['Dcrit'].sel(significance_level=alpha).values
    Dmean = ds['KS_D'].mean(dim='duration')
    single_map(Dmean,
        # title='Lilliefors test statistic (1979-2018, mean on $d$, $\\alpha=${})'.format(alpha),
        title='',
        cbar_label="Kolmogorov-Smirnov $D$ ($D_{{\mathrm{{crit}}}} = {:.2f}$)".format(Dcrit),
        center=Dcrit,
        reverse=True,
        fig_name='D_1979-2018_{}_dmean.png'.format(int(alpha*100)))


def fig_map_filliben(ds):
    """create maps of the Kolmogorov-Smirnov / Lilliefors test statistic
    """
    alpha = 0.05
    DURATION = 2
    r_crit = ds['filliben_crit'].sel(significance_level=alpha).values
    r_mean = ds['filliben_stat'].mean(dim='duration')
    single_map(r_mean,
        # title='Lilliefors test statistic (1979-2018, mean on $d$, $\\alpha=${})'.format(alpha),
        title='',
        cbar_label="Filliben's $r$ ($r_{{\mathrm{{crit}}}} = {:.2f}$)".format(r_crit),
        center=r_crit,
        reverse=False,
        fig_name='filliben_1979-2018_{}_dmean.png'.format(int(alpha*100)))


def fig_maps_gev24h(ds):
    da_loc = ds['location'].sel(duration=24)
    da_scale = ds['scale'].sel(duration=24)
    print(da_loc)
    print(da_scale)
    multi_maps([da_loc, da_scale],
               ['Location $\mu$', 'Scale $\sigma$'],
               'Parameter value', 'gev_params_24h_1979-2018_twobars.png',
               onebar=False,
               bar_labels=[da_loc.attrs['units'], da_scale.attrs['units']]
               )


def fig_maps_gev_scaled24h(ds):
    da_loc = ds['gev_scaled'].sel(ci='estimate', duration=24, ev_param='location').rename('location')
    da_scale = ds['gev_scaled'].sel(ci='estimate', duration=24, ev_param='scale').rename('scale')
    da_shape = ds['gev_scaled'].sel(ci='estimate', duration=24, ev_param='shape').rename('shape')
    multi_maps([da_loc, da_scale, da_shape],
                ['Location $\mu$', 'Scale $\sigma$', 'Shape $\kappa$'],
                'Parameter value', 'gev_params_scaled_24h_1979-2018.png')

def fig_maps_r(ds):
    multi_maps(ds.sel(scaling_extent=b'daily'), ['location_line_rvalue', 'scale_line_rvalue'],
               ['Location', 'Scale'],
               '$r$', 'gumbel_r_daily.png', sqr=False)


def fig_maps_rsquared(ds):
    da_scaling = ds['gev_scaling'].sel(ci='estimate', scaling_param='rsquared')
    da_loc_r2 = da_scaling.sel(ev_param='location').rename('location')
    da_scale_r2 = da_scaling.sel(ev_param='scale').rename('scale')
    multi_maps([da_loc_r2, da_scale_r2],
               ['Location', 'Scale'],
               '$r^2$', 'gev_r2.png', sqr=False)


def fig_maps_spearman(ds):
    da_scaling = ds['gev_scaling'].sel(ci='estimate', scaling_param='spearman')
    da_loc_rho = da_scaling.sel(ev_param='location').rename('location')
    da_scale_rho = da_scaling.sel(ev_param='scale').rename('scale')
    multi_maps([da_loc_rho, da_scale_rho],
               ['Location', 'Scale'],
               '$\\rho$', 'gev_rho.png', sqr=False, robust=False)


def fig_scaling_differences_all(ds_era, ds_midas, fig_name):
    quantiles = [0.025, 0.5, 0.975]

    ds_era_sel = ds_era.sel(latitude=ds_midas['latitude'],
                            longitude=postprocessing.convert_lon(ds_midas['longitude']),
                            method='nearest')

    q_extent_dict = {
        'Whole world': postprocessing.get_quantile_dict(quantiles,
                                         ERA5=(ds_era, ['latitude', 'longitude']),
                                         ),
        'MIDAS stations': postprocessing.get_quantile_dict(quantiles,
                                            ERA5=(ds_era_sel, ['station']),
                                            MIDAS=(ds_midas, ['station'])
                                            ),
        }

    # Flatten the dict
    q_list = []
    for extent, df_dict in q_extent_dict.items():
        for param, df_list in df_dict.items():
            q_list.append({'extent':extent,
                           'param':param,
                           'df_list':df_list})

    col_num = len(q_extent_dict)
    row_num = 2
    fig, axes = plt.subplots(row_num, col_num, figsize=(6, 3),
                             sharey='row', sharex=True)

    ylim = {'location': [-0.25, 0.75], 'scale': [-0.6, 1.1]}
    # param in rows, extent in columns
    for row_num, (param, ax_row) in enumerate(zip(['location', 'scale'], axes)):
        for extent, ax in zip(q_extent_dict, ax_row):
            df_list = [i['df_list'] for i in q_list
                        if i['extent'] == extent
                        and i['param'] == param][0]
            plot_scaling_differences(param, df_list, ax, ylim=False)
            if row_num == 0:
                ax.set_title(extent)

    # set ticks
    for ax in axes.flat:
        set_logd_xticks(ax)

    # add a big axes, hide frame
    # fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axes
    # plt.tick_params(labelcolor='none', top=False,
    #                 bottom=False, left=False, right=False)
    # plt.grid(False)
    # ylabel = '$\log_{{10}}( d^{{\eta}} / (\mu, \sigma) )$'
    # plt.ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, left=0.15, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name), bbox='tight')
    plt.close()


def plot_scaling_differences(param, df_list, ax, ylim=False):
    """
    Plot quantiles of differences
    """
    param_symbols = {'location': '\mu_d', 'scale': '\sigma_d'}
    gradient_symbols = {'location': '\\alpha', 'scale': '\\beta'}
    intercept_symbols = {'location': 'a', 'scale': 'b'}
    colors = {'ERA5': C_PRIMARY_1, 'MIDAS': C_PRIMARY_2,
              'MIDAS mean': C_PRIMARY_2, 'MIDAS stations': 'grey'}

    for source, df in df_list:
        quantiles = df.columns.values
        q_min = quantiles[0]
        q_max = quantiles[-1]
        title = '{} ${}$'.format(param.title(), param_symbols[param])
        fill_label = '{:.0%} of {}'.format(q_max-q_min, source)
        df[0.5].plot(logx=True, logy=False, ax=ax,
                        linewidth=1, color=colors[source], zorder=10,
                    #  title=title,
                        label='{} median'.format(source))
        ax.fill_between(df.index, df[q_min], df[q_max],
                        facecolor=colors[source], alpha=0.2, zorder=1,
                        label=fill_label)
    ax.axhline(0, linewidth=0.1, color='0.5')
    # ax.axvline(24, linewidth=0.1, color='0.5')
    if ylim:
        ax.set_ylim(ylim[param])
    # ax.set_yticks([ 0.0, 0.5])
    set_logd_xticks(ax)
    ax.set_xlabel('$d$ (hours)')
    # ylabel = '$\log_{{10}}( {i}d^{g} / {p} )$'.format(g=gradient_symbols[param],
    #                                                   p=param_symbols[param],
    #                                                   i=intercept_symbols[param])
    ylabel = "${i}d^{g} - {p}$".format(g=gradient_symbols[param],
                                       p=param_symbols[param],
                                       i=intercept_symbols[param])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))



def fig_scaling_ratio_map(ds):
    # scaling_ratio = ds['scaling_ratio'].sel(scaling_extent=b'all')
    alpha = ds['gev_scaling'].sel(scaling_param='slope', ci='estimate', ev_param='location')
    beta = ds['gev_scaling'].sel(scaling_param='slope', ci='estimate', ev_param='scale')
    scaling_ratio = alpha / beta
    single_map(scaling_ratio,
               title="",
               cbar_label='$\\alpha / \\beta$',
               center=1.0,
               fig_name='scaling_ratio1979-2018.png')


def fig_scaling_gradients_maps(ds):
    multi_maps([ds['alpha'], ds['beta']],
               ['Location scaling gradient $\\alpha$', 'Scale scaling gradient $\\beta$'],
               'Parameter value', 'scaling_gradients_1979-2018_twobars.png',
               onebar=False,
               bar_labels=['$\\alpha$', '$\\beta$'])


def fig_scaling_intersects_maps(ds):
    multi_maps([ds['a'], ds['b']],
               ['a', 'b'],
               'Parameter value', 'scaling_intersects_1979-2018_twobars.png',
               onebar=False,
               bar_labels=['a', 'b'])

def fig_scaling_gradients_ratio_maps(ds):
    """print maps of the ratio between the regression slopes obtain from daily and all durations
    """
    ds_daily = ds.sel(scaling_extent=b'daily')
    ds_all = ds.sel(scaling_extent=b'all')
    ds['loc_ratio'] = ds_all['location_line_slope'] / ds_daily['location_line_slope']
    ds['scale_ratio'] = ds_all['scale_line_slope'] / ds_daily['scale_line_slope']
    multi_maps(ds,
               var_names=['loc_ratio', 'scale_ratio'],
               disp_names=['$\\alpha_{all} / \\alpha_{daily}$',
                           '$\\beta_{all} / \\beta_{daily}$'],
               value_name="Ratio", fig_name='scaling_gradients_ratio.png',
               sqr=False, center=.5, reverse=True)


def fig_scaling_hexbin(ds):
    da1 = ds['gev_scaling'].sel(scaling_param='slope', ci='estimate', ev_param='location')
    da2 = ds['gev_scaling'].sel(scaling_param='slope', ci='estimate', ev_param='scale')
    hexbin(da1, da2,
           '$\\alpha$', '$\\beta$',
           'scaling_gradients_hexbin.png')


def table_r_sigcount(ds, alpha, dim):
    """Count the number of points with p > alpha (i.e. not significant).
    """
    ds_r = ds[['location_line_pvalue', 'scale_line_pvalue']].sel(scaling_extent=[
                                                                 b'daily', b'all'])
    print(ds_r.where(ds_r > alpha).count(dim=dim).load())


def table_count_noGEVfit(ds):
    da_loc = ds['gev'].sel(ci='estimate', ev_param='location')
    num_null = (~np.isfinite(da_loc)).sum(dim=['latitude', 'longitude'])
    print(num_null.load())


def table_count_noscalingfit(ds):
    da_scaling_slope = ds['gev_scaling'].sel(ci='estimate', ev_param=['location', 'scale'], scaling_param='slope')
    num_null = np.isnan(da_scaling_slope).sum(dim=['latitude', 'longitude'])
    print(num_null.load())


def table_rsquared_quantiles(ds, dim):
    ds_rsquared = ds['gev_scaling'].sel(ci='estimate', ev_param=['location', 'scale', 'shape'], scaling_param='rsquared')
    q =  ds_rsquared.load().quantile([0.01, 0.05], dim=dim)
    print(q)


def table_gof_quantiles(ds, dim):
    q_levels = [0.001, 0.01, 0.05, 0.1, 0.9, 0.95,0.99,0.999]

    q_D =  ds['KS_D'].load().quantile(q_levels)
    print("Dcrit: {}".format(ds['Dcrit'].load()))
    print(q_D)
    q_r = ds['filliben_stat'].load().quantile(q_levels)
    print("r_crit: {}".format(ds['filliben_crit'].load()))
    print(q_r)


def table_gof_count(ds):
    """count the number of cells x durations without good fit.
    """
    n_all = ds['filliben_stat'].count().load()
    nofit = (ds['filliben_stat'] < ds['filliben_crit'].sel(significance_level=0.05))
    n_nofit = xr.where(nofit, 1., np.nan).count().load()
    print(n_nofit/n_all)


def table_sizing_values(ds, coords, d):
    """Values over for the culvert sizing exercise.
    """
    ds_sel = ds.sel(latitude=coords[0],
                    longitude=postprocessing.convert_lon(coords[1]),
                    method='nearest').load()
    ds_sel = ds_sel.sel(ci=['estimate', '0.025', '0.975'],
                        duration=d,
                        ev_param=['location', 'scale'])
    # fitted
    da_fitted = ds_sel['gev']
    print(da_fitted)
    print(ds_sel['gev_scaled'])
    print(ds_sel['gev_scaling'].sel(scaling_param=['slope', 'intercept']))


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
            site_lon = postprocessing.convert_lon(ds_s['longitude'].values)
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


def fig_midas_mean(ds_pairs, fig_name):
    dict_df = postprocessing.ds_to_df(ds_pairs, 'pair')
    # # From seaborn colorblind
    # C = ['#0173b2', '#de8f05', '#029e73', '#cc78bc']
    C = {'ERA5': C_PRIMARY_1, 'MIDAS mean': C_PRIMARY_2,
         'MIDAS s1': 'grey','MIDAS s2': 'grey'}
    M = {'ERA5': 'o', 'MIDAS mean': 'v',
         'MIDAS s1': 'x','MIDAS s2': 'x'}
    # MS = {'ERA5': 4, 'MIDAS mean': 5,
    #       'MIDAS s1': 2,'MIDAS s2': 2}

    linesyles = {
        '{}_{}': dict(linestyle='None', linewidth=0, marker='o', markersize=4,
                      label='{}'),
        # '{}_{}_lr_all': dict(linestyle='dashed', linewidth=1.5, marker=None, markersize=0,
        #                      label='{} (all)'),
        '{}_{}_lr_daily': dict(linestyle='-', linewidth=1., marker=None, markersize=0,
                               label='{} (daily)'),
    }

    ax_list = []
    col_num = 2
    row_num = len(dict_df) + 1
    fig_size = (3.5*col_num, 2.2*row_num)
    # fig = plt.figure(figsize=fig_size)
    # ax_num = 1
    fig, axes = plt.subplots(row_num, col_num, figsize=fig_size,
                             sharey=False, sharex=True)
    for row_idx, ((pair_name, df), ax_row) in enumerate(zip(dict_df.items(), axes[:-1])):
        print(pair_name)
        for param_name, ax in zip(['location', 'scale'], ax_row):
            ax_list.append(ax)
            for source in ds_pairs['source'].values:
                for col_base, style in linesyles.items():
                    col = col_base.format(source, param_name)
                    if source.startswith('MIDAS s'):
                        if 'lr' in col:
                            linestyle = 'dashed'
                            marker = None
                            markersize = None
                            if source == 'MIDAS s1':
                                label = 'MIDAS stations (daily)'
                            else:
                                label = ''
                        else:
                            linestyle = 'none'
                            marker = 'x'
                            linewidth = .5
                            markersize = 2
                            if source == 'MIDAS s1':
                                label = 'MIDAS stations'
                            else:
                                label = ''
                    else:
                        linestyle = style['linestyle']
                        linewidth = style['linewidth']
                        marker = M[source]
                        markersize = style['markersize']
                        label = style['label'].format(source)
                    df[col].plot(ax=ax, label=label,
                                loglog=True,
                                linestyle=linestyle, linewidth=linewidth,
                                markersize=markersize, marker=marker,
                                color=C[source])
                    # except KeyError:
                    #     continue

            if row_idx == 0:
                ax.set_title(param_name)
            if param_name == 'location':
                ax.set_ylabel('{}'.format(pair_name))
            lines, labels = ax.get_legend_handles_labels()
            ax.set_xlabel('$d$ (hours)')

    ## Difference plot ##
    # build the dict of ds for calculating quantiles
    sites_list = []
    dict_quantiles = {}
    for s in ds_pairs.source.values:
        ds_source = ds_pairs.sel(source=s)
        if str(s).startswith('MIDAS s'):
            sites_list.append(ds_source)
        elif str(s).startswith('MIDAS'):
            dict_quantiles['MIDAS mean'] = (ds_source, ['pair'])
        elif str(s).startswith('ERA'):
            dict_quantiles['ERA5'] = (ds_source, ['pair'])
    ds_stations = xr.concat(sites_list, dim='pair')
    dict_quantiles['MIDAS stations'] = (ds_stations, ['pair'])
 
    quantiles = [0.01, 0.5, 0.99]
    q_dict = get_quantile_dict(quantiles, **dict_quantiles)

    # param in rows, extent in columns
    for param, df_list, ax_diff in zip(['location', 'scale'], q_dict.values(), axes[-1]):
        # ax_diff = fig.add_subplot(row_num, col_num, ax_num)
        plot_scaling_differences(param, df_list, ax_diff)
        # ax_diff.set_title(param)
        ax_list.append(ax_diff)

    # set ticks
    for ax in ax_list:
        set_logd_xticks(ax)

    lgd = fig.legend(lines, labels, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.1, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def scatter_intensity(da_i, fig_name):
    """
    """
    df_i = da_i.to_dataframe()
    # Split intensities in two columns, one for each source
    da_list = []
    for source in da_i['source'].values:
        series_i = df_i.xs(source, level='source').rename(columns={'intensity':source})
        da_list.append(series_i)
    df_i_s = pd.concat(da_list, axis='columns').reset_index()

    # Plot some durations on facetgrid
    dur_sel = [1,6,24,240]
    df_i_s = df_i_s.loc[np.in1d(df_i_s['duration'], dur_sel)]
    fg = sns.FacetGrid(df_i_s, sharex=False, sharey=False, row='T', col='duration')
    fg = fg.map(plt.plot, 'ERA5_scaled_rlm', 'MIDAS', color=C_PRIMARY_2, label='ERA5 scaled LTS')
    fg = fg.map(plt.scatter, 'ERA5_scaled', 'MIDAS', color=C_PRIMARY_2, label='ERA5 scaled')
    fg = fg.map(plt.plot, 'ERA5_rlm', 'MIDAS', color=C_PRIMARY_1, label='ERA5 LTS')
    fg = fg.map(plt.scatter, 'ERA5', 'MIDAS', color=C_PRIMARY_1, label='ERA5')
    for ax in fg.axes.flatten():
        lines, labels = ax.get_legend_handles_labels()
    fig = plt.gcf()
    lgd = fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.04, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_ARF(da, fig_name):
    width = 7
    aspect = 1.2
    col_wrap = 3
    height = (6/aspect)/col_wrap
    da = da.drop(['MIDAS', 'ERA5', 'ERA5_scaled'], dim='source')
    df = da.to_dataset(dim='source').to_dataframe().reset_index()
    # Plot on facetgrid
    fg = sns.FacetGrid(df, sharex=True, sharey=True, col='T', col_wrap=col_wrap, aspect=aspect, height=height)
    fg = fg.map(plt.axhline, y=1, color='0.8', linewidth=1.).set(xscale = 'log')
    fg = fg.map(plt.plot, 'duration', 'ERA5_scaled_rlm', color=C_PRIMARY_2).set(xscale = 'log')
    fg = fg.map(plt.plot, 'duration', 'ERA5_rlm', color=C_PRIMARY_1).set(xscale = 'log')
    for ax in fg.axes.flatten():
        set_logd_xticks(ax)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def intensities_errors(da):
    sns.set_style("ticks")
    sns.set_context("paper")
    width = 6.5
    aspect = 1.5
    col_wrap = 3
    height = (6/aspect)/col_wrap
    da = da.drop(['MIDAS', 'ERA5_rlm', 'ERA5_scaled_rlm'], dim='source')
    df = da.to_dataset(dim='source').to_dataframe().reset_index()
    print(df.head())
    # Plot on facetgrid
    fg = sns.FacetGrid(df, sharex=True, sharey=True, col='T', col_wrap=col_wrap, aspect=aspect, height=height)
    fg = fg.map(plt.plot, 'duration', 'ERA5_scaled', color=C_PRIMARY_2, label='PXR-4').set(xscale = 'log')
    fg = fg.map(plt.fill_between, 'duration', 'ERA5_scaled_ci_l', 'ERA5_scaled_ci_h',
                color=C_PRIMARY_2, alpha=0.2, linewidth=0, label='95% of PXR-4').set(xscale = 'log')
    fg = fg.map(plt.plot, 'duration', 'ERA5', color=C_PRIMARY_1, label='PXR-2').set(xscale = 'log')
    fg = fg.map(plt.fill_between, 'duration', 'ERA5_ci_l', 'ERA5_ci_h',
                color=C_PRIMARY_1, alpha=0.2, linewidth=0, label='95% of PXR-2').set(xscale = 'log')
    return fg


def plot_intensities_AE(da, ylabel, fig_name):
    fg = intensities_errors(da)
    fg.set_ylabels(ylabel)
    for ax in fg.axes.flatten():
        set_logd_xticks(ax)
        lines, labels = ax.get_legend_handles_labels()
    fig = plt.gcf()
    lgd = fig.legend(lines, labels, loc='lower center', ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.22, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_intensities_errors_percent(da, ylabel, fig_name):
    dur_min = 1
    dur_max = 360
    da_sel = da.sel(duration=slice(dur_min, dur_max))
    print(da_sel)
    fg = intensities_errors(da_sel)

    fg = fg.map(plt.axhline, y=0, color='0.8', linewidth=0.5, zorder=0)
    # plot +/1 20% error band
    err_band = 0.2
    # fg = fg.map(plt.axhline, y=err_band, color='0.8', linewidth=1., linestyle='dashed', zorder=0)
    # fg = fg.map(plt.axhline, y=-err_band, color='0.8', linewidth=1., linestyle='dashed', zorder=0,
    #             label='$\pm${:.0%} error band'.format(err_band))
    # polish plot
    fg.set_ylabels(ylabel)
    for ax in fg.axes.flatten():
        lines, labels = ax.get_legend_handles_labels()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        set_logd_xticks(ax, dur_min, dur_max)
    fig = plt.gcf()
    lgd = fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.22, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_scaling_intensity_error(ds, dim, fig_name):
    # calculate intensity for given duration and return period
    da_list = []
    for source in ['gev','gev_scaled']:
        # 95% CI
        da_gev = ds[source].sel(ci=['estimate', '0.025', '0.975'])
        i_list = []
        for T in [2,10,50,100,500,1000]:
            intensity = postprocessing.estimate_intensity(da_gev, ds['n_obs'], T)
            intensity = intensity.expand_dims('T')
            intensity.coords['T'] = [T]
            i_list.append(intensity)
        da_i = xr.concat(i_list, dim='T').rename(source)
        da_list.append(da_i)
    ds_i = xr.merge(da_list)

    # Computation of MPE
    orig = ds_i['gev']
    scaled = ds_i['gev_scaled']
    pe = (scaled - orig) / orig
    mpe_i = pe.mean(dim=dim).rename('mpe_i').sel(ci='estimate').load()
    # print(mpe_i)
    print(mpe_i.min())
    print(mpe_i.max())

    # Plot
    sns.set_style("ticks")
    sns.set_context("paper")
    width = 4
    aspect = 2
    height = width / aspect
    df = mpe_i.to_dataframe().reset_index()
    # print(df.head())
    ax = sns.lineplot(data=df, x='duration', y='mpe_i', style='T', color='0.3', ci=None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_ylabel('MPE')
    ax.set_xscale('log')
    set_logd_xticks(ax)
    fig = plt.gcf()
    lines, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(lines, labels, loc='upper right', ncol=1)
    ax.get_legend().remove()
    fig.set_size_inches(width, height)
    plt.tight_layout()
    plt.subplots_adjust(right=.75, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_hyetographs(ds_era, ds_midas, station_num, start, end, fig_name):
    ds_midas = ds_midas.sel(station=station_num).reset_coords(['latitude', 'longitude'])
    ds_era = ds_era.sel(latitude=ds_midas['latitude'],
                        longitude=postprocessing.convert_lon(ds_midas['longitude']),
                        method='nearest')
    precip_era = ds_era['precipitation'].rename('precip').sortby('time')
    precip_midas = ds_midas['prcp_amt'].rename('precip').rename({'end_time': 'time'}).sortby('time')
    # precip = xr.merge([precip_era, precip_midas])
    df_era = precip_era.sel(time=slice(start, end)).to_dataframe()
    df_midas = precip_midas.sel(time=slice(start, end)).to_dataframe()
    station_name = df_midas['src_name'][0].decode('utf8')
    # plot
    ax = df_era['precip'].plot(zorder=10, label='ERA5')
    df_midas['precip'].plot(ax=ax, zorder=0, label='MIDAS')
    ax.set_ylabel("P (mm/h)")
    ax.set_title("Station #{} '{}'".format(station_num, station_name))
    fig = plt.gcf()
    lines, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(lines, labels, loc='lower center', ncol=2)
    plt.subplots_adjust(bottom=.22, wspace=None, hspace=None)
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def plot_IDF(ds, coords, return_periods, fig_name, title=''):
    """Plot IDF curves and their uncertainties at given return periods
    """
    dur_min = 1
    dur_max = 48
    # Select the cells above the station
    ds_sel = ds.sel(latitude=coords[0],
                    longitude=postprocessing.convert_lon(coords[1]),
                    method='nearest'
                    ).drop(['longitude', 'latitude']).sel(duration=slice(dur_min, dur_max))
    # calculate intensity for given return periods
    da_list = []
    for T in return_periods:
        intensity = postprocessing.estimate_intensity(ds_sel['gev'], ds['n_obs'], T)
        intensity = intensity.expand_dims('T')
        intensity.coords['T'] = [T]
        da_list.append(intensity)
    da_i = xr.concat(da_list, dim='T')
    # Save the CI as variables
    ds_i = da_i.to_dataset(dim='ci')
    df = ds_i.to_dataframe().reset_index()

    # Plot
    sns.set(style="ticks", context='paper')
    sns.set_palette(sns.color_palette("viridis", len(return_periods)))
    fig_size = (4, 3)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for rt in return_periods:
        df_T = df.loc[df['T'] == rt].dropna()
        # plot estimate
        df_T.plot(x='duration', y='estimate', ax=ax, linewidth=0.1,
                       label="{} years".format(rt),
                       )
        # plot CI
        ax.fill_between(df_T['duration'], df_T['0.025'], df_T['0.975'],
                alpha=0.25, zorder=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Duration (hours)')
    ax.set_ylabel('Intensity (mm/h)')
    ax.set_title(title)
    set_logd_xticks(ax, dur_min, dur_max)
    lines, labels = ax.get_legend_handles_labels()
    # print(lines)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def scatter_ams(ds, study_sites, fig_title, fig_name):
    assert len(study_sites) == 2
    ams = postprocessing.combine_ds_per_site(study_sites, ds_cont={'ERA5': ds})['annual_max'].sel(source='ERA5', duration=1, drop=True)
    df_ams = ams.to_pandas().transpose()
    print(df_ams.iloc[:, 0])
    # spearman rho
    rho = helper.spearman_rho(df_ams.iloc[:, 0], df_ams.iloc[:, 1])
    print(rho)
    # plot
    ax = df_ams.plot.scatter(0, 1)
    ax.set_title(fig_title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fig_name))
    plt.close()


def paired_test(ds):
    """paired difference tests between alpha and beta
    """
    alpha = ds['alpha'].values.flatten()
    beta = ds['beta'].values.flatten()
    stat, pvalue = scipy.stats.wilcoxon(alpha, beta, zero_method='pratt')
    print(stat, pvalue)
    stat, pvalue = scipy.stats.ttest_rel(alpha, beta, nan_policy='omit')
    print(stat, pvalue)


def main():
    ds_era = xr.open_zarr(os.path.join(config.data_dir['era5'], config.era5_results))
    # ds_pxr2 = xr.open_dataset(os.path.join('../data', PXR2))
    # ds_pxr4 = xr.open_dataset(os.path.join('../data', PXR4))
    # print(ds_pxr4)
    # Drop Gibraltar
    ds_midas = xr.open_zarr(os.path.join(config.data_dir['midas'], config.midas_results))
    ds_midas = ds_midas.drop([1585], dim='station')
    # print(ds_era)

    # era_precip = xr.open_zarr(os.path.join(DATA_DIR, ERA_PRECIP_FILE))
    # midas_precip = xr.open_zarr(os.path.join(config.data_dir['midas'], config.hourly_filename['midas'])
    # plot_hyetographs(era_precip, midas_precip, station_num=23, start='1980-01', end='1980-12', fig_name='hyetographs_23_1980.pdf')
    # plot_hyetographs(era_precip, midas_precip, station_num=23, start='1979', end='2018', fig_name='hyetographs_23_1979-2018.pdf')
    # plot_hyetographs(era_precip, midas_precip, station_num=23, start='2017', end='2018', fig_name='hyetographs_23_2017-2018.pdf')

    # paired_test(ds_pxr4)

    ##########
    # In Manuscript
    ##########

    # fig_map_KS(ds_era)
    # fig_map_filliben(ds_era)

    # fig_maps_gev24h(ds_pxr2)
    # fig_scaling_gradients_maps(ds_pxr4)
    # fig_scaling_intersects_maps(ds_pxr4)

    # fig_maps_rsquared(ds_era)

    # fig_scaling_differences_all(ds_era, ds_midas, 'scaling_diff_all_1979-2018.pdf')
    # fig_scaling_ratio_map(ds_era)
    # fig_scaling_hexbin(ds_era)

    # fig_gauges_map('midas_gauges_map.pdf')

    ds_combined = postprocessing.combine_ds_per_site(STUDY_SITES, ds_cont={'ERA5': ds_era})
    plot_scaling_per_site(ds_combined, 'sites_scaling_1979-2018.pdf')
    # print(ds_combined['gev_scaled'].sel(duration=[1, 24], station='Jakarta', ci=['estimate', '0.025', '0.975'], ev_param='location').load())

    # ds_i = postprocessing.estimate_intensities_errors(ds_era, ds_midas)
    # plot_intensities_errors_percent(ds_i['mdpe'], 'MdPE', 'MdPE_intensities.pdf')
    # plot_scaling_intensity_error(ds_era, dim=['longitude', 'latitude'], fig_name='MPE_intensities_scaled.pdf')

    # plot_IDF(ds_era, STUDY_SITES['Vienna'], [2, 10, 100], 'IDF_Vienna.png', title='IDF for Vienna from PXR-2')

    ##############

    # fig_maps_gev_scaled24h(ds_era)
    # table_count_noGEVfit(ds_era)
    # table_count_noscalingfit(ds_era)
    # table_rsquared_quantiles(ds_era, dim=['longitude', 'latitude'])
    # table_gof_count(ds_era)

    # fig_maps_spearman(ds_era)

    # ds_pairs = prepare_midas_mean(ds_era, ds_midas, ds_midas_pairs)
    # fig_midas_mean(ds_pairs, 'midas_mean.pdf')

    # table_sizing_values(ds_era, STUDY_SITES['Jakarta'], d=2)

    # scatter_intensity(ds_i['intensity'], 'scatter_intensity.pdf')
    # plot_ARF(ds_i['arf'], 'arf_scaling.pdf')
    # plot_intensities_AE(ds_i['mae'], 'MAE (mm/h)', 'MAE_intensities.pdf')
    # plot_intensities_errors_percent(ds_i['mape'], 'MAPE', 'MAPE_intensities.pdf')
    # postprocessing.adequacy(ds_i['mape'], threshold=.2)


    # era2018_precip = xr.open_zarr('/home/lunet/gylc4/geodata/ERA5/yearly_zarr/2018.zarr')
    # precip_2018_max = era2018_precip['precipitation'].max(dim='time')
    # single_map(precip_2018_max,
    #            title="",
    #            cbar_label='intensity (mm/h)',
    #            fig_name='2018_max.png')

    # scatter_ams(ds_era, STUDY_SITES, fig_title='Comparison in 24h annual maxima (mm/h), 1979-2018', fig_name='ams_scatter.pdf')

if __name__ == "__main__":
    sys.exit(main())
