# -*- coding: utf8 -*-
import sys
import os
import datetime

import xarray as xr
import toml

DATA_DIR = '../data'
SOURCE = 'era5_2000-2017_precip_complete.zarr'
METDATA = 'metadata.toml'

with open(METDATA, 'r') as toml_file:
    metadata = toml.loads(toml_file.read())

# output
PXR2 = f"pxr2-{metadata['common']['version']}.nc"
PXR4 = f"pxr4-{metadata['common']['version']}.nc"


def gen_pxr2(ds_source):
    # Select the relevant variables
    ds_pxr2 = ds_source[['location', 'scale', 'A2', 'A2_crit']]
    # Dataset description
    ds_pxr2.attrs.update(metadata['common'])
    ds_pxr2.attrs.update(metadata['pxr2'])

    for coordinate in ds_pxr2.coords:
        ds_pxr2.coords[coordinate].attrs.update(metadata['coords'][coordinate])
    for var in ds_pxr2.data_vars:
        try:
            ds_pxr2.data_vars[var].attrs.update(metadata['data_vars'][var])
        except KeyError:
            pass
        print(ds_pxr2.data_vars[var])
    print(ds_pxr2)
    return ds_pxr2


def main():
    source_path = os.path.join(DATA_DIR, SOURCE)
    ds_source = xr.open_zarr(source_path).sel(gumbel_fit=b'scipy', drop=True)
    ds_pxr2 = gen_pxr2(ds_source)
    # ds_pxr2.to_netcdf(os.path.join(DATA_DIR, PXR2))


if __name__ == "__main__":
    sys.exit(main())
