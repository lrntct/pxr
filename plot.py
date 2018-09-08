# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy as ctpy
import matplotlib.pyplot as plt

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_2000-2012_precip.zarr'
ANNUAL_FILE = 'era5_2000-2012_precip_pearsonr.zarr'
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


def gumbel_map(ds, var_names, disp_names, value_name, fig_name, sqr=False):
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


def plot_sites(ds, sites, fig_name):
    # extract sites values from the dataset as pandas dataframes
    dict_df = {}
    scaling_coeffs_series = []
    for site_name, site_coord in sites.items():
        ds_sel = ds.sel(latitude=site_coord[0], longitude=site_coord[1], method='nearest').drop(['latitude', 'longitude'])
        scaling_coeffs = ['loc_lr_intercept', 'loc_lr_slope', 'scale_lr_intercept', 'scale_lr_slope', 'scaling_pearsonr']
        scaling_coeffs_series.append(ds_sel[scaling_coeffs].to_array(name=site_name).to_series())
        dict_df[site_name] = ds_sel[['loc_final', 'scale_final']].to_dataframe()
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
        linesyles = {'loc_final': dict(linestyle='None', marker='o', color='#1b9e77', label='Location $\mu$'),
                     'loc_lr': dict(linestyle='solid', marker=None, color='#1b9e77', label='$d^{\eta(\mu)}$'),
                     'scale_final': dict(linestyle='None', marker='o', color='#d95f02', label='Scale $\sigma$'),
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

        txt = "$\eta(\mu)$ = {:.2f}\n$\eta(\sigma)$ = {:.2f}\nPearson's r = {:.2f}".format(
                eta['loc'], eta['scale'],
                df_scaling_coeffs.loc['scaling_pearsonr', site_name])
        ax.text(0.05, 0.15, txt, horizontalalignment='left', backgroundcolor='white',
                verticalalignment='center', transform=ax.transAxes, size=10)
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


def main():
    ds = xr.open_zarr(os.path.join(DATA_DIR, ANNUAL_FILE))

    # print(ds[['scale_prov', 'scale_final']].loc[{'duration':24, 'latitude':0, 'longitude':slice(0, 1)}].load())
    # gumbel_map(ds, ['loc_final', 'scale_final'],
    #             ['Location $\mu$', 'Scale $\sigma$'],
    #             'Parameter value', 'gumbel_params.png')
    # gumbel_map(ds, ['loc_lr_slope', 'scale_lr_slope'],
    #            ['Location-Duration relationship', 'Scale-duration relationship'],
    #            '$\eta$', 'gumbel_scaling.png')
    # print(ds.where(ds['log_location'].isnull()).count().compute())

    # gumbel_map(ds, ['prov_lr_rvalue', 'final_lr_rvalue', 'loc_lr_rvalue', 'scale_lr_rvalue'],
    #            ['First Gumbel estimate', 'Final Gumbel fitting', 'Location-Duration', 'Scale-Duration'],
    #            '$r^2$', 'gumbel_r2.png', sqr=True)

    plot_sites(ds, STUDY_SITES, 'sites_scaling.png')

    # single_map(ds['scaling_pearsonr'],
    #            title="$d^{\eta(\mu)}$ - $d^{\eta(\sigma)}$ correlation",
    #            cbar_label='Pearson correlation coefficient',
    #            fig_name='pearsonr.pdf')

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
