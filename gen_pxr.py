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
        try:
            ds_pxr2.coords[coordinate].attrs.update(metadata['coords'][coordinate])
        except KeyError:
            pass
    for var in ds_pxr2.data_vars:
        try:
            ds_pxr2.data_vars[var].attrs.update(metadata['data_vars'][var])
        except KeyError:
            pass
    return ds_pxr2


def gen_pxr4(ds_source):
    # Select the relevant variables
    ds_pxr4 = ds_source[['location_line_intercept', 'location_line_slope', 'location_line_rvalue',
                         'scale_line_intercept', 'scale_line_slope', 'scale_line_rvalue']]
    ds_pxr4['a'] = 10**ds_pxr4['location_line_intercept']
    ds_pxr4['b'] = 10**ds_pxr4['scale_line_intercept']
    ds_pxr4['location_r2'] = ds_pxr4['location_line_rvalue'] * ds_pxr4['location_line_rvalue']
    ds_pxr4['scale_r2'] = ds_pxr4['scale_line_rvalue'] * ds_pxr4['scale_line_rvalue']
    for d in ['location_line_intercept', 'scale_line_intercept',
              'location_line_rvalue', 'scale_line_rvalue']:
        del ds_pxr4[d]
    rename_dict = {'location_line_slope': 'alpha',
                   'scale_line_slope': 'beta'}
    ds_pxr4.rename(rename_dict, inplace=True)
    # Dataset description
    ds_pxr4.attrs.update(metadata['common'])
    ds_pxr4.attrs.update(metadata['pxr4'])

    for coordinate in ds_pxr4.coords:
        try:
            ds_pxr4.coords[coordinate].attrs.update(metadata['coords'][coordinate])
        except KeyError:
            pass
    for var in ds_pxr4.data_vars:
        try:
            ds_pxr4.data_vars[var].attrs.update(metadata['data_vars'][var])
        except KeyError:
            pass
    return ds_pxr4


def main():
    source_path = os.path.join(DATA_DIR, SOURCE)
    ds_source = xr.open_zarr(source_path).sel(gumbel_fit=b'scipy', scaling_extent=b'all', drop=True)
    print(ds_source)
    ds_pxr2 = gen_pxr2(ds_source)
    # ds_pxr2.to_netcdf(os.path.join(DATA_DIR, PXR2))
    ds_pxr4 = gen_pxr4(ds_source)
    print(ds_pxr4)
    ds_pxr4.to_netcdf(os.path.join(DATA_DIR, PXR4))


if __name__ == "__main__":
    sys.exit(main())
