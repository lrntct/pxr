# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime
import math

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry
import seaborn as sns
import cartopy as ctpy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt


DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_2000-2012_precip.zarr'
ANNUAL_FILE = 'era5_2000-2017_precip_scaling.zarr'
ERA_MULTIDAILY_FILE = 'era5_2000-2012_precip_scaling_multidaily.zarr'
GAUGES_FILE = '../data/GHCN/ghcn.nc'
GHCN_ANNUAL_FILE = '../data/GHCN/ghcn_2000-2017_precip_annual_max.nc'
MIDAS_ANNUAL_FILE = '../data/MIDAS/midas_2000-2017_precip_scaling.nc'
HADISD_ANNUAL_FILE = '../data/HadISD/hadisd_2000-2017_precip_scaling.nc'
PLOT_DIR = '../plot'

EXTRACT = dict(latitude=slice(45, -45),
               longitude=slice(0, 180))

# Coordinates of study sites
STUDY_SITES = {'Kampala': (0.317, 32.616),
               'Kisumu': (0.1, 34.75)}


def single_map(da, cbar_label, title, fig_name):
    """https://cbrownley.wordpress.com/2018/05/15/visualizing-global-land-temperatures-in-python-with-scrapy-xarray-and-cartopy/
    """
    plt.figure(figsize=(8, 5))
    ax_p = plt.gca(projection=ctpy.crs.Robinson(), aspect='auto')
    ax_p.coastlines(linewidth=.3, color='black')
    da.plot.imshow(ax=ax_p, transform=ctpy.crs.PlateCarree(),
                   robust=True, cmap='viridis',
                   cbar_kwargs=dict(orientation='horizontal', label=cbar_label))
    plt.title(title)
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def multi_maps(ds, var_names, disp_names, value_name, fig_name, sqr=False):
    crs = ctpy.crs.Robinson()
    sel = ds.loc[{'duration':24}]
    # reshape variables as dimension
    da_list = []
    for var_name in var_names:
        if sqr == True:
            da_sel = xr.ufuncs.square(sel[var_name]).expand_dims('param')
        else:
            da_sel = sel[var_name].expand_dims('param')
        da_sel.coords['param'] = [var_name]
        da_list.append(da_sel)
    da = xr.concat(da_list, 'param')
    da.attrs['long_name'] = value_name  # Color bar title
    # Actual plot
    p = da.plot(col='param', col_wrap=1,
                transform=ctpy.crs.PlateCarree(),
                aspect=ds.dims['longitude'] / ds.dims['latitude'],
                cmap='viridis', robust=True, extend='both',
                subplot_kws=dict(projection=crs)
                )
    for ax, disp_name in zip(p.axes.flat, disp_names):
        ax.coastlines(linewidth=.5, color='black')
        ax.set_title(disp_name)
    # plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def prepare_scaling_per_site(ds_era, ds_gauges, name_col, ds_era_multidaily=None):
    """
    """
    # ORGANIZE DATA #
    scaling_coeffs = ['loc_lr_intercept', 'loc_lr_slope',
                      'scale_lr_intercept', 'scale_lr_slope',
                      'scaling_pearsonr', 'scaling_spearmanr', 'scaling_ratio']
    # Set station coordinates to station name
    ds_gauges.coords['station'] = ds_gauges[name_col]

    # dict of coordinates. Used to select sites on ERA5 dataset
    sites = {n:(lat,lon) for n, lat, lon in zip(ds_gauges[name_col].values,
                                                ds_gauges['latitude'].values,
                                                ds_gauges['longitude'].values)
             if n != 'HONOLULU OBSERVATORY 702.2, HI US'}  # Wrong values
    # print(sites)
    # extract sites values from the ERA5 dataset as pandas dataframes
    dict_df = {}
    dict_scaling_coeffs = {}
    for site_name, site_coord in list(sites.items()):
        # print(site_name)
        if site_name not in [b'BRIZE NORTON', b'LITTLE RISSINGTON',
                             b'LARKHILL', b'BOSCOMBE DOWN']:
            continue
        print(site_name)
        ds_era_sel = (ds_era.sel(latitude=site_coord[0],
                           longitude=site_coord[1],
                           method='nearest')
                      .drop(['latitude', 'longitude']))
        df_era = ds_era_sel[['loc_loaiciga', 'scale_loaiciga']].to_dataframe()
        df_era.rename(columns={'loc_loaiciga': 'era_loc',
                               'scale_loaiciga': 'era_scale'},
                      inplace=True)
        if ds_era_multidaily:
            ds_era_multidaily_sel = (ds_era_multidaily
                                    .sel(latitude=site_coord[0],
                                        longitude=site_coord[1],
                                        method='nearest')
                                    .drop(['latitude', 'longitude']))
            era_m_scaling_coeffs = ds_era_multidaily_sel[scaling_coeffs].to_array(name='ERA5_MULTIDAILY').to_series()

        ds_gauges_sel = ds_gauges[['loc_loaiciga', 'scale_loaiciga']]
        df_gauges = ds_gauges_sel.sel(station=site_name).to_dataframe()
        df_gauges.rename(columns={'loc_loaiciga': 'gauges_loc',
                                  'scale_loaiciga': 'gauges_scale'},
                         inplace=True)
        for c in ['latitude', 'longitude', name_col, 'station', 'code']:
            try:
                df_gauges.drop([c], axis=1, inplace=True)
            except KeyError:
                pass

        dict_df[site_name] = pd.concat([df_era, df_gauges], axis=1, sort=False)

        era_scaling_coeffs = ds_era_sel[scaling_coeffs].to_array(name='ERA5').to_series()
        # print(era_scaling_coeffs)
        gauges_scaling_coeff = (ds_gauges
                                .sel(station=site_name)[scaling_coeffs]
                                .to_array(name='gauges')
                                .to_series())
        for c in ['station', name_col, 'code', 'latitude', 'longitude']:
            try:
                gauges_scaling_coeff.drop([c], inplace=True)
            except KeyError:
                pass
        # print(gauges_scaling_coeff)
        if ds_era_multidaily:
            dict_scaling_coeffs[site_name] = pd.concat([era_scaling_coeffs,
                                                        era_m_scaling_coeffs,
                                                        gauges_scaling_coeff],
                                                    axis=1, sort=False)
        else:
            dict_scaling_coeffs[site_name] = pd.concat([era_scaling_coeffs,
                                                        gauges_scaling_coeff],
                                                    axis=1, sort=False)
    return dict_df, dict_scaling_coeffs


def plot_scaling_per_site(dict_df, dict_scaling_coeffs, fig_name):
    """
    """
    col_num = 2
    row_num = math.ceil(len(dict_df)/2)
    fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True, figsize=(4*col_num,3*row_num))
    for (site_name, df), ax in zip(dict_df.items(), axes.flat):
        # print(site_name)
        # calculate regression lines
        df_scaling_coeffs = dict_scaling_coeffs[site_name]
        for p in ['loc', 'scale']:
            inter_row = '{}_lr_intercept'.format(p)
            slope_row = '{}_lr_slope'.format(p)
            for source in ['ERA5', 'ERA5_MULTIDAILY', 'gauges']:
                try:
                    intercept = df_scaling_coeffs.loc[inter_row, source]
                    slope = df_scaling_coeffs.loc[slope_row, source]
                    regline_col = '{}_{}_lr'.format(source, p)
                    df[regline_col] = 10**intercept * df.index**slope
                except KeyError:
                    continue
        # plot
        # color-blind safe qualitative palette from colorbrewer2
        c1a = '#6ba9ca'
        c1b = '#1f78b4'
        c2a = '#89c653'
        c2b = '#33a02c'
        linesyles = {
            'era_loc': dict(linestyle='None', linewidth=0, marker='o', markersize=2, color=c1a, label='Location $\mu$ (ERA5)'),
            'ERA5_loc_lr': dict(linestyle='dashed', linewidth=1., marker=None, markersize=0, color=c1a, label='$d^{\eta(\mu)}$ (ERA5)'),
            'era_scale': dict(linestyle='None', linewidth=0, marker='o', markersize=2, color=c2a, label='Scale $\sigma$ (ERA5)'),
            'ERA5_scale_lr': dict(linestyle='dashed', linewidth=1., marker=None, markersize=0, color=c2a, label='$d^{\eta(\sigma)}$ (ERA5)'),
            'ERA5_MULTIDAILY_loc_lr': dict(linestyle='dotted', linewidth=1., marker=None, markersize=0, color=c1a, label='$d^{\eta(\mu)}$ (ERA5 daily)'),
            'ERA5_MULTIDAILY_scale_lr': dict(linestyle='dotted', linewidth=1., marker=None, markersize=0, color=c2a, label='$d^{\eta(\sigma)}$ (ERA5 daily)'),
            'gauges_loc': dict(linestyle='None', linewidth=0, marker='v', markersize=3, color=c1b, label='Location $\mu$ (gauges)'),
            'gauges_loc_lr': dict(linestyle='solid', linewidth=0.5, marker=None, markersize=0, color=c1b, label='$d^{\eta(\mu)}$ (gauges)'),
            'gauges_scale': dict(linestyle='None', linewidth=0, marker='v', markersize=3, color=c2b, label='Scale $\sigma$ (gauges)'),
            'gauges_scale_lr': dict(linestyle='solid', linewidth=0.5, marker=None, markersize=0, color=c2b, label='$d^{\eta(\sigma)}$ (gauges)')
                    }
        for col, styles in linesyles.items():
            try:
                df[col].plot(loglog=True, title=site_name.decode("utf-8"), ax=ax, label=styles['label'],
                        linestyle=styles['linestyle'], linewidth=styles['linewidth'],
                        markersize=styles['markersize'],
                        marker=styles['marker'], color=styles['color'])
            except KeyError:
                continue
        lines, labels = ax.get_legend_handles_labels()
        ax.set_xlabel('$d$ (hours)')
        ax.set_ylabel('$\mu, \sigma$')
        ax.set_xticks([1, 3, 6, 12, 24, 48, 120, 360])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # Text box
        # table_row = "{:<11s} {:>5.2f} {:>8.2f} {:>5.2f}\n"
        # txt = ["{:<11s} {:>5s} {:>8s} {:>5s}\n".format('Parameter', 'ERA5', 'ERA5 day', 'GHCN'),
        #        table_row.format('Loc. slope',
        #                         #'$\eta(\mu)$',
        #                         df_scaling_coeffs.loc['loc_lr_slope', 'ERA5'],
        #                         df_scaling_coeffs.loc['loc_lr_slope', 'ERA5_MULTIDAILY'],
        #                         df_scaling_coeffs.loc['loc_lr_slope', 'GHCN']),
        #        table_row.format('Scale slope',
        #                         #'$\eta(\sigma)$',
        #                         df_scaling_coeffs.loc['scale_lr_slope', 'ERA5'],
        #                         df_scaling_coeffs.loc['scale_lr_slope', 'ERA5_MULTIDAILY'],
        #                         df_scaling_coeffs.loc['scale_lr_slope', 'GHCN']),
        #        table_row.format('Slope ratio',
        #                         #'$\eta(\mu) / \eta(\sigma)$',
        #                         df_scaling_coeffs.loc['scaling_ratio', 'ERA5'],
        #                         df_scaling_coeffs.loc['scaling_ratio', 'ERA5_MULTIDAILY'],
        #                         df_scaling_coeffs.loc['scaling_ratio', 'GHCN'])
        #     ]
        # t = ax.text(0.01, 0.0, ''.join(txt), backgroundcolor='white',
        #             horizontalalignment='left', verticalalignment='bottom',
        #             transform=ax.transAxes, size=7, family='monospace'
        #             )
        # t.set_bbox(dict(alpha=0))  # force transparent background
    # plt.legend(lines, labels, loc='lower center', ncol=4)
    lgd = fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.2, wspace=None, hspace=None)
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
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
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def plot_gumbel_per_site(ds, sites, fig_name):
    DURATION = 6
    # extract sites values from the dataset as pandas dataframes
    dict_cdf = {}
    for site_name, site_coord in sites.items():
        dict_pvalues = {}
        ds_sel = ds.sel(latitude=site_coord[0],
                        longitude=site_coord[1],
                        duration=DURATION,
                        method='nearest').drop(['latitude', 'longitude', 'duration'])
        # Keep only wanted values as dataframe
        keep_col = ['estim_prob', 'analytic_prob_moments', 'analytic_prob_loaiciga', 'annual_max']
        df_cdf = ds_sel[keep_col].load().to_dataframe().set_index('annual_max', drop=True).sort_index()
        dict_cdf[site_name] = df_cdf

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8,4))
    for (site_name, df), ax in zip(dict_cdf.items(), axes):
        dict_cdf[site_name].plot(linestyle=None, marker='o', markersize=1.5,
                                 title=site_name, ax=ax, legend=False)
        lines, labels = ax.get_legend_handles_labels()
        ax.set_ylabel('Cumulative probability')
        txt = "Dcrit = {:.2f}".format(ds['Dcrit_5pct'].values)
        ax.text(0.65, 0.10, txt, horizontalalignment='left', backgroundcolor='white',
                verticalalignment='center', transform=ax.transAxes, size=10)
    fig.suptitle('Cumulative probability for a duration of {} hours'.format(DURATION))
    lgd_labels = ['Estimated CDF',
                  'Analytic CDF (method of moments)',
                  'Analytic CDF (iterative method)']
    lgd = fig.legend(lines, lgd_labels, loc='lower center', ncol=4)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=.24, wspace=None, hspace=None)
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
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
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def plot_gauges_map(ds, id_dim, fig_name, global_extent=True):
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
        ax_p = plt.gca(projection=ctpy.crs.Robinson(), aspect='auto')
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


    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def hexbin(ds, var1, var2, fig_name):
    x = ds[var1].values.flat
    y = ds[var2].values.flat
    xq = np.nanquantile(x, [0.01, 0.99])
    yq = np.nanquantile(y, [0.01, 0.99])
    xmin, xmax = xq[0], xq[1]
    ymin, ymax = xq[0], xq[1]

    fig, ax = plt.subplots(figsize=(4, 3))
    # ax.axis([xmin, xmax, ymin, ymax])
    hb = ax.hexbin(x, y, gridsize=20, extent=[xmin, xmax, ymin, ymax], mincnt=1)
    ax.set_xlabel('$\eta(\mu)$')
    ax.set_ylabel('$\eta(\sigma)$')
    ax.plot([xmin, xmax], [ymin, ymax], color='k', linewidth=0.5)
    norm = matplotlib.colors.Normalize(vmin=1000, vmax=12000)
    cb = plt.colorbar(hb, spacing='uniform', orientation='vertical',
                      label='# of cells', norm=norm, extend='both')
    # cb.set_label('# of cells')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()


def main():
    # ds_era = xr.open_zarr(os.path.join(DATA_DIR, ANNUAL_FILE))
    # print(ds_era)
    # print(ds_era[['ks_loaiciga', 'ks_moments']].load().quantile([0.95,0.99,0.999]))
    # ds_ghcn = xr.open_dataset(GHCN_ANNUAL_FILE)
    # print(ds_ghcn)
    ds_annual_midas = xr.open_dataset(MIDAS_ANNUAL_FILE)
    # ds_annual_hadisd = xr.open_dataset(HADISD_ANNUAL_FILE)
    # print(ds_annual_midas)

    # hexbin(ds_era, 'loc_lr_slope', 'scale_lr_slope',
    #        'scaling_gradients_hexbin.png')

    # ds_annual_ghcn = xr.open_zarr(GAUGES_ANNUAL_FILE)
    # ds_era_multidaily = xr.open_zarr(os.path.join(DATA_DIR, ERA_MULTIDAILY_FILE))
    # print(ds_annual_gauges.load())
    # print(ds_annual_midas.load())
    plot_gauges_map(ds_annual_midas, 'src_name', 'midas_gauges_map.pdf', global_extent=False)
    # plot_gauges_map(ds_ghcn, 'name', 'ghcn_gauges_map.pdf', global_extent=True)
    # plot_gauges_data(ds_ghcn, 2000, 2012 'gauges.png')

    # print(ds[['scale_prov', 'scale_final']].loc[{'duration':24, 'latitude':0, 'longitude':slice(0, 1)}].load())
    # multi_maps(ds_era, ['loc_loaiciga', 'scale_loaiciga'],
    #             ['Location $\mu$', 'Scale $\sigma$'],
    #             'Parameter value', 'gumbel_params_24h_2000-2017.png')
    # multi_maps(ds_era, ['loc_lr_slope', 'scale_lr_slope'],
    #            ['Location-Duration relationship', 'Scale-duration relationship'],
    #            '$\eta$', 'gumbel_scaling_2000-2017.png')
    # print(ds.where(ds_era['log_location'].isnull()).count().compute())

    # multi_maps(ds_era, ['prov_lr_rvalue', 'final_lr_rvalue', 'loc_lr_rvalue', 'scale_lr_rvalue'],
    #            ['First Gumbel estimate', 'Final Gumbel fitting', 'Location-Duration', 'Scale-Duration'],
    #            '$r^2$', 'gumbel_r2.png', sqr=True)

    # print((~np.isfinite(ds)).sum().compute())
    # plot_gumbel_per_site(ds_era, STUDY_SITES, 'sites_gumbel.png')
    # dict_df, dict_scaling_coeffs = prepare_scaling_per_site(ds_era, ds_annual_ghcn, ds_era_multidaily)
    # dict_df, dict_scaling_coeffs = prepare_scaling_per_site(ds_era, ds_annual_midas, 'src_name')
    # plot_scaling_per_site(dict_df, dict_scaling_coeffs, 'sites_scaling_midas_select_2000-2017.pdf')

    # single_map(ds_era['scaling_pearsonr'],
    #            title="$d^{\eta(\mu)}$ - $d^{\eta(\sigma)}$ correlation",
    #            cbar_label='Pearson correlation coefficient',
    #            fig_name='pearsonr.png')

    # single_map(ds_era['scaling_ratio'],
    #            title="Parameter scaling ratio",
    #            cbar_label='$\eta(\mu) / \eta(\sigma)$',
    #            fig_name='scaling_ratio2000-2017.png')

    # single_map(ds_era['scaling_spearmanr'],
    #        title="Parameter scaling correlation",
    #        cbar_label="Spearman's $\\rho$",
    #        fig_name='spearmanr.png')

    # multi_maps(ds_era, ['ks_loaiciga', 'ks_moments'],
    #            ['Fitting accuracy of the iterative method (d=24h)', 'Fitting accuracy of the method of moments (d=24h)'],
    #            "Kolmogorov-Smirnov's D", 'fitting_accuracy.png', sqr=False)

    # hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
    # hourly = xr.open_zarr(hourly_path)
    # plot_hourly(hourly, STUDY_SITES, 'hourly.png')

    # hourly maxima
    # hourly_max = hourly['precipitation'].max('time')
    # print(hourly_max)
    # single_map(hourly_max,
    #            title="Max hourly precipitation 2000-2012 (ERA5)",
    #            cbar_label='Precipitation rate (mm/hr)',
    #            fig_name='hourly_max.png')

    block_one = ['BRIZE NORTON', 'LITTLE RISSINGTON']
    block_two = ['LARKHILL', 'BOSCOMBE DOWN']

if __name__ == "__main__":
    sys.exit(main())
