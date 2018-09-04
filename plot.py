# -*- coding: utf8 -*-
import matplotlib
matplotlib.use("Agg")

import sys
import os
import datetime

import numpy as np
import xarray as xr
import seaborn as sns
import cartopy as ctpy
import matplotlib.pyplot as plt

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
ANNUAL_FILE_GUMBEL = 'era5_2000-2012_precip_gradient.nc'


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


def gumbel_map(ds, var_names, fig_name, sqr=False):
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
    p = da.plot(col='param', col_wrap=1,
                transform=ctpy.crs.PlateCarree(),
                aspect=ds.dims['longitude'] / ds.dims['latitude'],
                cmap='viridis', robust=True, extend='both',
                subplot_kws=dict(projection=crs)
                )
    for ax in p.axes.flat:
        ax.coastlines(linewidth=.5, color='black')
    plt.savefig(fig_name)
    plt.close()


def main():
    gumbel_coeff = xr.open_dataset(os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL))
    gumbel_map(gumbel_coeff, ['loc_final', 'scale_final'], 'gumbel_params.png')
    gumbel_map(gumbel_coeff, ['loc_lr_slope', 'scale_lr_slope'], 'gumbel_scaling.png')
    gumbel_map(gumbel_coeff, ['prov_lr_rvalue', 'final_lr_rvalue', 'loc_lr_rvalue', 'scale_lr_rvalue'], 'gumbel_r2.png', sqr=True)





if __name__ == "__main__":
    sys.exit(main())
