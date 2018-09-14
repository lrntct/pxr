#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import multiprocessing as mp

import cdsapi
# from ecmwfapi import ECMWFDataServer


FORMAT = 'grib'
PRODUCT = 'reanalysis-era5-single-levels'
VARIABLE = [#'mean_total_precipitation_rate',
            'total_precipitation']
TYPE = 'reanalysis'
MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
DAY = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
       '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
TIME = ['00:00','01:00','02:00','03:00','04:00','05:00',
        '06:00','07:00','08:00','09:00','10:00','11:00',
        '12:00','13:00','14:00','15:00','16:00','17:00',
        '18:00','19:00','20:00','21:00','22:00','23:00']
START_YEAR = 2015
END_YEAR = 2015


def dl_cdsapi(year_month):
    c = cdsapi.Client()
    # try:
    query = {'variable': VARIABLE,
             'product_type': TYPE,
             'year': str(year_month[0]),
             'month': year_month[1],
             'day': DAY,
             'time': TIME,
             'format':FORMAT}
    r = c.retrieve(PRODUCT, query)
    filename = '{}-{}.{}'.format(year_month[0], year_month[1], FORMAT)
    r.download(filename)
    # except:
    #     pass


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
    pool = mp.Pool(10)
    years = range(START_YEAR, END_YEAR+1)
    year_month = [(y, m) for y in years for m in MONTH]
    # print(year_month)
    pool.map(dl_cdsapi, year_month)

if __name__ == "__main__":
    sys.exit(main())
