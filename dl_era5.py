#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import os
import shutil
import multiprocessing as mp
import datetime
import urllib3.exceptions

import xarray as xr
import cdsapi
import humanize
import requests

# import read_grib

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/ensemble'

# FORMAT = 'grib'
FORMAT = 'netcdf'
PRODUCT = 'reanalysis-era5-single-levels'
VARIABLE = [
            #'mean_total_precipitation_rate'
            'total_precipitation'
            ]
# TYPE = 'reanalysis'
TYPE = 'ensemble_members'
MONTH = [str(i+1).zfill(2) for i in range(12)]
DAY = [str(i+1).zfill(2) for i in range(31)]
TIME = ['{}:00'.format(i).zfill(5) for i in range(24)]
START_YEAR = 1979
END_YEAR = 1979


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
    # convert to zarr and delete grib
    ds = xr.open_dataset(disk_url)
    print(ds)
    # read_grib.grib2zarr(disk_url, DATA_DIR)
    # os.remove(disk_url)


def main():
    pool = mp.Pool(16)
    years = range(START_YEAR, END_YEAR+1)
    year_month = [(y, m) for y in years for m in MONTH]
    # print(year_month)
    pool.map(dl_cdsapi, year_month)

if __name__ == "__main__":
    sys.exit(main())
