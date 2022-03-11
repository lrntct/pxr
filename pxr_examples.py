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
import os
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import cartopy as ctpy
import xarray as xr
import numpy as np

DATA_DIR = '../data'
VERSION = '2.1.0'
PXR2 = f"pxr2-{VERSION}.nc"
PXR4 = f"pxr4-{VERSION}.nc"

LOC_MEX = (19.43, -99.13)  # Latitude and longitude of Mexico City, as an example
SHAPE = -0.114  # PXR v2.1.0 is using a fixed value of the GEV shape parameter

def gumbel_quantile(T, loc, scale):
    """Return quantile (i.e, intensity) for a given return period T in years
    """
    y = -np.log(-np.log(1 - 1/T))
    return loc + scale * y


def gev_quantile_nonzero(T, loc, scale, shape):
    """Return quantile (i.e, intensity) for a given return period T in years
    Consider an EV type II if shape<0
    """
    y = (1 - (-np.log(1 - 1/T))**shape) / shape
    return loc + scale * y


def gev_quantile(T, loc, scale, shape):
    """T: return period in years
    """
    return xr.where(shape == 0,
                    gumbel_quantile(T, loc, scale),
                    gev_quantile_nonzero(T, loc, scale, shape))


def get_single_intensity(ds_pxr2, return_period, duration, lat, lon):
    """For a given location, duration and return period,
    return the intensity as a float.
    """
    ds_select = ds_pxr2.sel(latitude=lat, longitude=lon, method='nearest')
    ds_select = ds_select.sel(duration=duration)
    i = gev_quantile(return_period, ds_select['location'],
                     ds_select['scale'], SHAPE)
    return i.item()


def get_point_idf(ds_pxr2, lat, lon, return_periods):
    """At a given location, calculate the IDF curves for
    a given list of return periods.
    Return a DataArray of intensities.
    """
    ds_select = ds_pxr2.sel(latitude=lat, longitude=lon, method='nearest')
    da_list = []
    for T in return_periods:
        intensities = gev_quantile(T, ds_select['location'],
                     ds_select['scale'], SHAPE)
        # print(intensity)
        intensities = intensities.expand_dims('T')
        intensities.coords['T'] = [T]
        da_list.append(intensities)
    return xr.concat(da_list, dim='T').rename('intensities')


def get_global_intensities(ds_pxr2, duration, return_period):
    """For a given return period and duration,
    return a DataArray of global intensities.
    """
    ds_select = ds_pxr2.sel(duration=duration)
    intensities = gev_quantile(return_period, ds_select['location'],
                               ds_select['scale'], SHAPE)
    return intensities


def plot_point_idf(da_idf, fig_name):
    """Plot IDF curves at a single point.
    """
    fig_size = (4, 3)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for rt in da_idf['T']:
        rt = rt.item()
        da_select = da_idf.sel(T=rt).squeeze()
        df = da_select.to_dataframe().reset_index()
        # plot intensity estimate
        df.plot(x='duration', y='intensities', ax=ax, linewidth=0.1,
                       label="T = {} years".format(rt),
                       )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Duration (hours)')
    ax.set_ylabel('Intensity (mm/h)')
    ax.set_title('IDF curves')
    # set_logd_xticks(ax, dur_min, dur_max)
    lines, labels = ax.get_legend_handles_labels()
    # print(lines)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def plot_intensities_map(da, fig_name):
    """Plot a map of intensities.
    """
    plt.figure(figsize=(8, 5))
    ax_p = plt.axes(projection=ctpy.crs.EqualEarth(), aspect='auto')
    da.plot.imshow(ax=ax_p, transform=ctpy.crs.PlateCarree(),
                   robust=True, cmap='viridis', center=False,
                   cbar_kwargs=dict(orientation='horizontal', label='Intensities (mm/h)'))
    ax_p.coastlines(linewidth=.5, color='black')
    plt.title('Intensities')
    plt.savefig(fig_name)
    plt.close()


def main():
    ds_pxr2 = xr.open_dataset(os.path.join(DATA_DIR, PXR2))
    i = get_single_intensity(ds_pxr2, 1000, 1, LOC_MEX[0], LOC_MEX[1])
    print(i)
    da_idf = get_point_idf(ds_pxr2, LOC_MEX[0], LOC_MEX[1], [10, 50, 100])
    plot_point_idf(da_idf, 'point_idf.pdf')
    global_i = get_global_intensities(ds_pxr2, 24, 100)
    plot_intensities_map(global_i, 'i_map.pdf')


if __name__ == "__main__":
    sys.exit(main())
