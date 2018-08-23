import os
import sys
import shutil
import multiprocessing as mp

import requests

BASE_URL = 'https://catalogue.ceh.ac.uk/datastore/eidchub/33604ea0-c238-4488-813d-0ad9ab7c51ca/GB/daily/'
BASE_NAME = 'CEH_GEAR_daily_GB_{}.nc'
LOCAL_DIR = '/media/laurent/2TBHDD/GeoData/CEH_GEAR'


def runner(year):
    file_name = BASE_NAME.format(year)
    print(file_name)
    local_file = os.path.join(LOCAL_DIR, file_name)
    url = os.path.join(BASE_URL, file_name)
    with requests.get(url, stream=True) as r:
        with open(local_file, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def main():
    pool = mp.Pool(4)
    r = pool.map_async(runner, range(1950, 2016))
    r.wait()
    # pool.close()


if __name__ == "__main__":
    sys.exit(main())
