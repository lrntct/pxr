# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry
import seaborn as sns
import cartopy as ctpy
import matplotlib.pyplot as plt

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_2000-2012_precip.zarr'
ANNUAL_FILE = 'era5_2000-2012_precip_scaling.zarr'
GAUGES_FILE = '../data/GHCN/ghcn.nc'
GAUGES_ANNUAL_FILE = '../data/GHCN/ghcn_2000-2012_precip_scaling.zarr'
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


def scaling_per_site(ds, sites, fig_name):
    # extract sites values from the dataset as pandas dataframes
    dict_df = {}
    scaling_coeffs_series = []
    for site_name, site_coord in sites.items():
        ds_sel = ds.sel(latitude=site_coord[0], longitude=site_coord[1], method='nearest').drop(['latitude', 'longitude'])
        scaling_coeffs = ['loc_lr_intercept', 'loc_lr_slope',
                          'scale_lr_intercept', 'scale_lr_slope',
                          'scaling_pearsonr', 'scaling_spearmanr', 'scaling_ratio']
        scaling_coeffs_series.append(ds_sel[scaling_coeffs].to_array(name=site_name).to_series())
        dict_df[site_name] = ds_sel[['loc_loaiciga', 'scale_loaiciga']].to_dataframe()
    df_scaling_coeffs = pd.DataFrame(scaling_coeffs_series).T
    # print(df_scaling_coeffs)
    # print(dict_df['Kisumu'])

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8,4))
    for (site_name, df), ax in zip(dict_df.items(), axes):
        # calculate regression lines
        eta = {}
        for p in ['loc', 'scale']:
            inter_col = '{}_lr_intercept'.format(p)
            slope_col = '{}_lr_slope'.format(p)
            intercept = df_scaling_coeffs.loc[inter_col, site_name]
            slope = df_scaling_coeffs.loc[slope_col, site_name]
            eta[p] = slope
            regline_col = '{}_lr'.format(p)
            df[regline_col] = 10**intercept * df.index**slope
        # plot
        linesyles = {'loc_loaiciga': dict(linestyle='None', marker='o', color='#1b9e77', label='Location $\mu$'),
                     'loc_lr': dict(linestyle='solid', marker=None, color='#1b9e77', label='$d^{\eta(\mu)}$'),
                     'scale_loaiciga': dict(linestyle='None', marker='o', color='#d95f02', label='Scale $\sigma$'),
                     'scale_lr': dict(linestyle='solid', marker=None, color='#d95f02', label='$d^{\eta(\sigma)}$'),
                    }
        for col, styles in linesyles.items():
            df[col].plot(loglog=True, title=site_name, ax=ax, markersize=2, label=styles['label'],
                    linestyle=styles['linestyle'], marker=styles['marker'], color=styles['color'])

        lines, labels = ax.get_legend_handles_labels()
        ax.set_xlabel('$d$ (hours)')
        ax.set_ylabel('$\mu, \sigma$')
        ax.set_xticks([1, 3, 6, 12, 24, 48, 120, 360])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        txt = ("$\eta(\mu)$ = {:.2f}\n"
               "$\eta(\sigma)$ = {:.2f}\n"
               "Pearson's r = {:.2f}\n"
               "Spearman's $\\rho$ = {:.2f}\n"
               "$\eta(\mu) / \eta(\sigma)$  = {:.2f}").format(
                        eta['loc'], eta['scale'],
                        df_scaling_coeffs.loc['scaling_pearsonr', site_name],
                        df_scaling_coeffs.loc['scaling_spearmanr', site_name],
                        df_scaling_coeffs.loc['scaling_ratio', site_name]
                        )
        t = ax.text(0.05, 0.20, txt, horizontalalignment='left', backgroundcolor='white',
                verticalalignment='center', transform=ax.transAxes, size=10)
        t.set_bbox(dict(alpha=0))
    # plt.legend(lines, labels, loc='lower center', ncol=4)
    lgd = fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(bottom=.24, wspace=None, hspace=None)
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


def plot_gauges_map(ds, fig_name):
    """convert dataset in geopandas DataFrame
    plot it
    """
    ds_sel = ds.sel(duration=24, year=2000)['annual_max']
    # print(ds_sel)
    df = ds_sel.to_dataframe().set_index('code', drop=True).drop(axis=1, labels=['annual_max', 'year', 'duration'])

    df['geometry'] = [shapely.geometry.Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    # print(gdf)

    plt.figure(figsize=(8, 5))
    ax_p = plt.gca(projection=ctpy.crs.Robinson(), aspect='auto')
    ax_p.set_global()
    ax_p.coastlines(linewidth=.3, color='black')
    gdf.plot(ax=ax_p, transform=ctpy.crs.PlateCarree())
    plt.savefig(os.path.join(PLOT_DIR, fig_name))
    plt.close()

def main():
    ds_era = xr.open_zarr(os.path.join(DATA_DIR, ANNUAL_FILE))
    # print(ds_era)
    # ds_gauges = xr.open_dataset(GAUGES_FILE)
    ds_annual_gauges = xr.open_zarr(GAUGES_ANNUAL_FILE)
    # print(ds_annual_gauges.load())
    plot_gauges_map(ds_annual_gauges, 'gauges_map.png')
    # plot_gauges_data(ds_gauges, 2000, 2012 'gauges.png')

    # print(ds[['scale_prov', 'scale_final']].loc[{'duration':24, 'latitude':0, 'longitude':slice(0, 1)}].load())
    # multi_maps(ds_era, ['loc_final', 'scale_final'],
    #             ['Location $\mu$', 'Scale $\sigma$'],
    #             'Parameter value', 'gumbel_params.png')
    # multi_maps(ds_era, ['loc_lr_slope', 'scale_lr_slope'],
    #            ['Location-Duration relationship', 'Scale-duration relationship'],
    #            '$\eta$', 'gumbel_scaling.png')
    # print(ds.where(ds_era['log_location'].isnull()).count().compute())

    # multi_maps(ds_era, ['prov_lr_rvalue', 'final_lr_rvalue', 'loc_lr_rvalue', 'scale_lr_rvalue'],
    #            ['First Gumbel estimate', 'Final Gumbel fitting', 'Location-Duration', 'Scale-Duration'],
    #            '$r^2$', 'gumbel_r2.png', sqr=True)

    # print((~np.isfinite(ds)).sum().compute())
    # plot_gumbel_per_site(ds_era, STUDY_SITES, 'sites_gumbel.png')
    # scaling_per_site(ds_era, STUDY_SITES, 'sites_scaling.png')

    # single_map(ds_era['scaling_pearsonr'],
    #            title="$d^{\eta(\mu)}$ - $d^{\eta(\sigma)}$ correlation",
    #            cbar_label='Pearson correlation coefficient',
    #            fig_name='pearsonr.png')

    # single_map(ds_era['scaling_ratio'],
    #            title="Parameter scaling ratio",
    #            cbar_label='$\eta(\mu) / \eta(\sigma)$',
    #            fig_name='scaling_ratio.png')

    # single_map(ds_era['scaling_spearmanr'],
    #        title="Parameter scaling correlation",
    #        cbar_label="Spearman's $\\rho$",
    #        fig_name='spearmanr.png')

    # multi_maps(ds_era, ['ks_loaiciga', 'ks_moments'],
    #            ['Fitting accuracy for iterative method (d=6h)', 'Fitting accuracy for mthod of moments (d=6h)'],
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


if __name__ == "__main__":
    sys.exit(main())
