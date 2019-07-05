# -*- coding: utf8 -*-

"""Here goes the information expected to be changed by the user
"""

import os

analysis = dict(
            # Data on which the analysis is performed. 'era5' or 'midas'
            source='era5',
            # Start and end years for preprocessing and analysis
            start=1979,
            end=2018,
            # Number of bootstrap samples
            bootstrap_samples=1000,
            # Set to None to estimate the GEV shape parameter. Negative: Fréchet
            # ev_shape=-0.114,
            ev_shape=None,
)

# Where the data are stored on disk
data_dir = {'era5': '/media/drive2/ERA5/',
            'midas': '../data/MIDAS/',
}

# for MIDAS, a file. For ERA5, a dir of yearly files
hourly_filename = {'era5': 'yearly_zarr',
                   'midas': 'midas_{}-{}_precip_select.zarr'.format(analysis['start'], analysis['end'])}

plot = dict(dir='../plot',
            # Vector of location of station points
            station_map='../data/MIDAS/midas.gpkg',
            # A vector basemap for the station map
            base_map='../data/ne_land_10m.gpkg',
)


## Automaticly generate filenames ##

result_basename = '{}_{}-{}_ams{}.zarr'

path_ams = os.path.join(data_dir[analysis['source']],
                        result_basename.format(analysis['source'], analysis['start'], analysis['end'], ''))
path_ranked = os.path.join(data_dir[analysis['source']],
                           result_basename.format(analysis['source'], analysis['start'], analysis['end'], '_ranked'))
path_gev = os.path.join(data_dir[analysis['source']],
                        result_basename.format(analysis['source'], analysis['start'], analysis['end'], '_gev_kappa'))
path_gof = os.path.join(data_dir[analysis['source']],
                        result_basename.format(analysis['source'], analysis['start'], analysis['end'], '_gof_kappa'))

# Data files used to plot and generate PXR dataset
era5_results = result_basename.format('era5', analysis['start'], analysis['end'], '_gof')
midas_results = result_basename.format('midas', analysis['start'], analysis['end'], '_gof')
