import os
import sys
import shutil
import multiprocessing as mp

import requests

BASE_URL = 'http://hydrology.princeton.edu/data/hylkeb/MSWEP_V2.2_/global_3hourly_010deg/'
BASE_NAME = '{y}{m}.nc'
LOCAL_DIR = '/home/laurent/Documents/GeoData/MSWEP'
MONTHS = ['01','02','03','04','05','06','07','08','09','10','11','12']

def runner(year):
    for month in MONTHS:
        file_name = BASE_NAME.format(y=year, m=month)
        print(file_name)
        local_file = os.path.join(LOCAL_DIR, file_name)
        url = os.path.join(BASE_URL, file_name)
        with requests.get(url, stream=True) as r:
            with open(local_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


def main():
    pool = mp.Pool()
    r = pool.map(runner, range(1979, 2017))
    r.wait()
    # pool.close()


if __name__ == "__main__":
    sys.exit(main())
