#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import os
import shutil
import multiprocessing as mp
import datetime
import urllib3.exceptions

import cdsapi
import humanize
import requests
# from ecmwfapi import ECMWFDataServer

DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/monthly_grib_total_precip'

FORMAT = 'grib'
PRODUCT = 'reanalysis-era5-single-levels'
VARIABLE = [
            #'mean_total_precipitation_rate'
            'total_precipitation'
            ]
TYPE = 'reanalysis'
# TYPE = 'ensemble_members'
MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
DAY = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
       '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
TIME = ['00:00','01:00','02:00','03:00','04:00','05:00',
        '06:00','07:00','08:00','09:00','10:00','11:00',
        '12:00','13:00','14:00','15:00','16:00','17:00',
        '18:00','19:00','20:00','21:00','22:00','23:00']
START_YEAR = 1980
END_YEAR = 1999


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
            try:
                resp = requests.get(url, stream=True, timeout=(5, 5),
                                    headers={'Range': f'bytes={dl_size}-'})
                shutil.copyfileobj(resp.raw, f)
            except (requests.exceptions.ReadTimeout, urllib3.exceptions.ReadTimeoutError) as e:
                continue
            dl_size = f.tell()

    # cds_r.download(filename)



# def dl_ecmwf(year):
#     server = ECMWFDataServer()
#     server.retrieve({
#         "class": "ea",
#         "dataset": "era5",
#         "expver": "1",
#         "stream": "oper",
#         "type": "an",
#         "levtype": "sfc",
#         "param": "165.128/166.128/167.128",
#         "date": "2016-01-01/to/2016-01-02",
#         "time": "00:00:00",
#         "step": "0",
#         "grid": "0.25/0.25",
#         "area": "75/-20/10/60",
#         "format": "netcdf",
#         "target": "test.nc"
#      })

def main():
    pool = mp.Pool(8)
    years = range(START_YEAR, END_YEAR+1)
    year_month = [(y, m) for y in years for m in MONTH]
    # print(year_month)
    pool.map(dl_cdsapi, year_month)

if __name__ == "__main__":
    sys.exit(main())
