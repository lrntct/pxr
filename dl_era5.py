#!/usr/bin/env python3
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
import shutil
import multiprocessing as mp
import datetime
import urllib3.exceptions

import xarray as xr
import zarr
import cdsapi
import humanize
import requests

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_dl'
ZARR_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_zarr'

FORMAT = 'netcdf'
PRODUCT = 'reanalysis-era5-single-levels'
VARIABLE = [
            'total_precipitation'
            ]
TYPE = 'reanalysis'
# TYPE = 'ensemble_members'
MONTH = [str(i+1).zfill(2) for i in range(12)]
DAY = [str(i+1).zfill(2) for i in range(31)]
TIME = ['{}:00'.format(i).zfill(5) for i in range(24)]
START_YEAR = 2018
END_YEAR = 2018

DTYPE = 'float32'
CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
ENCODING = {'precipitation': GEN_FLOAT_ENCODING}

def get_url(year, month):
    cds_client = cdsapi.Client()
    query = {'variable': VARIABLE,
             'product_type': TYPE,
             'year': year,
             'month': month,
             'day': DAY,
             'time': TIME,
             'format': FORMAT}
    cds_r = cds_client.retrieve(PRODUCT, query)
    url = cds_r.location
    file_size = int(cds_r.content_length)
    return url, file_size


def dl_cdsapi(year_month):
    """Download one month of data as a single file
    """
    # saved filename on disk
    filename = '{}-{}.{}'.format(year_month[0], year_month[1], FORMAT)
    # Make sure that the TCP request returns the right file size:
    size_diff = 1
    while size_diff != 0:
        url, file_size = get_url(str(year_month[0]), year_month[1])
        print(url)
        resp = requests.get(url, stream=True, timeout=(5, 5))
        total_length = int(resp.headers.get('content-length'))
        size_diff = file_size - total_length

    # Retry until the file is complete
    disk_url = os.path.join(DATA_DIR, filename)
    dl_size = 0
    while dl_size < file_size:
        with open(disk_url, 'ab') as f:
            dl_size = f.tell()
            print("{}, full size:{} DL starting at {}".format(filename,
                humanize.naturalsize(file_size),
                humanize.naturalsize(dl_size))
                )
            # If timeout, re-initiate the download
            headers = {'Range': 'bytes={}-'.format(dl_size)}
            try:
                resp = requests.get(url, stream=True, timeout=(5, 5),
                                    headers=headers)
                shutil.copyfileobj(resp.raw, f)
            except (requests.exceptions.ReadTimeout, urllib3.exceptions.ReadTimeoutError) as e:
                continue
            dl_size = f.tell()
    # convert from m to mm (per hr), save to zarr and delete netcdf
    ds = xr.open_dataset(disk_url).chunk(CHUNKS).rename({'tp': 'precipitation'})
    ds = ds * 1000
    print(ds)
    zarr_filename = filename = '{}-{}.zarr'.format(year_month[0], year_month[1])
    out_file_path = os.path.join(ZARR_DIR, zarr_filename)
    ds.to_zarr(out_file_path, mode='w', encoding=ENCODING)
    os.remove(disk_url)


def main():
    pool = mp.Pool(12)
    years = range(START_YEAR, END_YEAR+1)
    year_month = [(y, m) for y in years for m in MONTH]
    pool.map(dl_cdsapi, year_month)


if __name__ == "__main__":
    sys.exit(main())
