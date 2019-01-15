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


def apply_common_metadata(ds):
    ds.attrs.update(metadata['common'])
    ds.attrs['date'] = str(datetime.datetime.utcnow().date())
    for coordinate in ds.coords:
        try:
            ds.coords[coordinate].attrs.update(metadata['coords'][coordinate])
        except KeyError:
            pass
    for var in ds.data_vars:
        try:
            ds.data_vars[var].attrs.update(metadata['data_vars'][var])
        except KeyError:
            pass


def gen_pxr2(ds_source):
    # Select the relevant variables
    ds_pxr2 = ds_source.sel(scaling_extent='all', drop=True)[['location', 'scale', 'A2', 'A2_crit']]
    # Metadata
    apply_common_metadata(ds_pxr2)
    ds_pxr2.attrs.update(metadata['pxr2'])
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
    # Compute ratio
    ds_all = ds_source.sel(scaling_extent='all')
    ds_daily =  ds_source.sel(scaling_extent='daily')
    ds_pxr4['alpha_ratio'] = ds_all['location_line_slope'] / ds_daily['location_line_slope']
    ds_pxr4['beta_ratio'] = ds_all['scale_line_slope'] / ds_daily['scale_line_slope']
    # Metadata
    apply_common_metadata(ds_pxr4)
    ds_pxr4.attrs.update(metadata['pxr4'])
    return ds_pxr4


def write_datasets(ds_source):
    # print(ds_source)
    ds_pxr2 = gen_pxr2(ds_source)
    ds_pxr2.to_netcdf(os.path.join(DATA_DIR, PXR2))
    ds_pxr4 = gen_pxr4(ds_source)
    # print(ds_pxr4)
    ds_pxr4.to_netcdf(os.path.join(DATA_DIR, PXR4))


def test_datasets():
    for file_name in [PXR2, PXR4]:
        ds = xr.open_dataset(os.path.join(DATA_DIR, file_name))
        print(ds)
        for coordinate in ds.coords:
            print(ds.coords[coordinate])
        for var in ds.data_vars:
            print(ds.data_vars[var])


def main():
    source_path = os.path.join(DATA_DIR, SOURCE)
    ds_source = xr.open_zarr(source_path).sel(gumbel_fit=b'scipy', drop=True).sel(scaling_extent=[b'all', b'daily'])
    ds_source['scaling_extent'] = ds_source['scaling_extent'].astype(str)
    write_datasets(ds_source)
    test_datasets()


if __name__ == "__main__":
    sys.exit(main())
