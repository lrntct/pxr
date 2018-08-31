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
ANNUAL_FILE_GUMBEL = 'era5_precip_gumbel.nc'


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


def gumbel_map(ds):
    crs = ctpy.crs.Robinson()
    # fig, axes = plt.subplots(1, 2, figsize=(10, 6), subplot_kw=dict(projection=crs))

    sel = ds.loc[{'duration':6}]
    # reshape variables as dimension
    da_list = []
    for param in ['loc', 'scale']:
        var_name = '{}_final'.format(param)
        da_sel = ds[var_name].expand_dims('param')
        da_sel.coords['param'] = [param]
        da_list.append(da_sel)
    da = xr.concat(da_list, 'param')
    da_6 = da.loc[{'duration':6}]
    # print(da_6)
    p = da_6.plot(col='param', col_wrap=1,
                  transform=ctpy.crs.PlateCarree(),
                  aspect=ds.dims['longitude'] / ds.dims['latitude'],
                  robust=True, vmin=0, vmax=20, extend='max', cmap='viridis',
                  subplot_kws=dict(projection=crs)
                  )
    for ax in p.axes.flat:
        ax.coastlines(linewidth=.5, color='black')
    plt.savefig('gumbel.png')
    plt.close()


def main():
  gumbel_coeff = xr.open_dataset(os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL))
  gumbel_map(gumbel_coeff)

if __name__ == "__main__":
    sys.exit(main())
