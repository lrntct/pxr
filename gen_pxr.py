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

import xarray as xr
import numpy as np
import toml

SOURCE_PATH = '/home/laurent/Documents/era5_1979-2018_ams_gof.zarr'
DATA_DIR = '../data'
METADATA = 'metadata.toml'

with open(METADATA, 'r') as toml_file:
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


def extract_ci(ds, var):
    """Separate the ci estimate from the CI intervals.
    return a ds
    """
    params = ds[var].to_dataset(dim='ev_param')[['location', 'scale']]
    params_est = params.sel(ci='estimate', drop=True)
    params_ci = params.drop('estimate', dim='ci').rename({'location': 'location_ci',
                                                          'scale': 'scale_ci'})
    params_ci.coords['ci'] = params_ci.coords['ci'].astype(np.float32)
    return xr.merge([params_est, params_ci])


def gen_pxr2(ds_source):
    params = extract_ci(ds_source, 'gev')
    # Select the relevant variables
    var_keep = ['filliben_stat', 'filliben_crit', 'KS_D', 'Dcrit']
    ds_pxr2 = xr.merge([params, ds_source[var_keep]]).rename({'KS_D': 'D'})
    # Sort dimension orders of GoF variables
    ds_pxr2['filliben_stat'] = ds_pxr2['filliben_stat'].transpose(*ds_pxr2['location'].dims)
    ds_pxr2['D'] = ds_pxr2['D'].transpose(*ds_pxr2['location'].dims)
    # Metadata
    apply_common_metadata(ds_pxr2)
    ds_pxr2.attrs.update(metadata['pxr2'])
    return ds_pxr2


def gen_pxr4(ds_source):
    # Select the relevant variables
    ds_pxr4 = ds_source[['filliben_stat', 'filliben_crit', 'KS_D', 'Dcrit']].rename({'KS_D': 'D'})
    # Extract CI
    params = extract_ci(ds_source, 'gev_scaling')
    # Estimates
    ds_pxr4['a'] = 10**params['location'].to_dataset(dim='scaling_param')['intercept']
    ds_pxr4['alpha'] = params['location'].to_dataset(dim='scaling_param')['slope']
    ds_pxr4['b'] = 10**params['scale'].to_dataset(dim='scaling_param')['intercept']
    ds_pxr4['beta'] = params['scale'].to_dataset(dim='scaling_param')['slope']
    # CI
    ds_pxr4['a_ci'] = params['location_ci'].sel(scaling_param='intercept')
    ds_pxr4['alpha_ci'] = params['location_ci'].sel(scaling_param='slope')
    ds_pxr4['b_ci'] = params['scale_ci'].sel(scaling_param='intercept')
    ds_pxr4['beta_ci'] = params['scale_ci'].sel(scaling_param='slope')

    ds_pxr4['location_r2'] = params['location'].to_dataset(dim='scaling_param')['rsquared']
    ds_pxr4['scale_r2'] = params['scale'].to_dataset(dim='scaling_param')['rsquared']
    ds_pxr4 = ds_pxr4.drop('scaling_param')

    # Metadata
    apply_common_metadata(ds_pxr4)
    ds_pxr4.attrs.update(metadata['pxr4'])
    return ds_pxr4


def write_datasets(ds_source):
    ds_pxr2 = gen_pxr2(ds_source)
    ds_pxr2.to_netcdf(os.path.join(DATA_DIR, PXR2))

    ds_pxr4 = gen_pxr4(ds_source)
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
    ds_source = xr.open_zarr(SOURCE_PATH)
    print(ds_source)
    write_datasets(ds_source)
    test_datasets()


if __name__ == "__main__":
    sys.exit(main())
