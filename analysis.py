# -*- coding: utf8 -*-
import sys
import os
import copy
from datetime import datetime, timedelta
import csv

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import zarr
import scipy.stats


DATA_DIR = '/home/lunet/gylc4/geodata/ERA5'
HOURLY_FILE = 'era5_2000-2012_precip.zarr'
ANNUAL_FILE = 'era5_2000-2012_precip_annual_max.zarr'
ANNUAL_FILE_RANK = 'era5_2000-2012_precip_ranked.zarr'
ANNUAL_FILE_GUMBEL = 'era5_2000-2012_precip_gumbel.zarr'
ANNUAL_FILE_GRADIENT = 'era5_2000-2012_precip_gradient.zarr'
ANNUAL_FILE_PEARSONR = 'era5_2000-2012_precip_pearsonr.zarr'

LOG_FILENAME = 'Analysis_log_{}.csv'.format(str(datetime.now()))


# Annual max from Rob
KAMPALA_AMS = '/home/lunet/gylc4/Sync/papers/global rainfall freq analysis/data/rob_kampala.csv'
KISUMU_AMS = '/home/lunet/gylc4/Sync/papers/global rainfall freq analysis/data/rob_kisumu.csv'

# Coordinates of study sites
KAMPALA_COORD = (0.317, 32.616)
KISUMU_COORD = (0.1, 34.75)
# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32.5, 35))
EXTRACT = dict(latitude=slice(0, -5),
               longitude=slice(0, 5))

# Event durations in hours - has to be adjusted to temporal resolution for the moving window
DURATIONS = [i+1 for i in range(24)] + [i for i in range(24+6,48+6,6)] + [i*24 for i in [5,10,15]]
TEMP_RES = 1  # Temporal resolution in hours

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 8, 'longitude': 8}
ANNUAL_CHUNKS = {'year': -1, 'duration':10, 'latitude': 45*4, 'longitude': 45*4}  # 4 cells: 1 degree
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
ANNUAL_ENCODING = {'annual_max': GEN_FLOAT_ENCODING,
                   'duration': {'dtype': DTYPE},
                   'latitude': {'dtype': DTYPE},
                   'longitude': {'dtype': DTYPE}}


def step1_annual_maxs_of_roll_mean(ds, durations, temp_res):
    """for each rolling winfows size:
    compute the annual maximum of a moving mean
    return an array with the durations as a new dimension
    """
    annual_maxs = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip = ds.precipitation
        precip_roll_mean = precip.rolling(time=window_size).mean(dim='time')
        annual_max = precip_roll_mean.groupby('time.year').max(dim='time')
        annual_max.name = 'annual_max'
        da = annual_max.expand_dims('duration')
        da.coords['duration'] = [duration]
        annual_maxs.append(da)
    return xr.concat(annual_maxs, 'duration')


def slm_ols_wrapper(x, y):
    """call statsmodels OLS and make it look like scipy linregress
    ['slope', 'intercept', 'rsquared', 'pvalue', 'stderr']
    """
    X = sm.tools.tools.add_constant(x)
    ols = slm.OLS(y, X, missing='drop')
    results = ols.fit()
    return results.params[1], results.params[0], results.rsquared, results.pvalues[0], results.bse[0]


def linregress(ds, x, y, prefix, dims):
    """ds: xarray dataset
    x, y: name of variables to use for the regression
    prefix: to be added before the indivudual result names
    dims: dimension on which to carry out the regression
    """
    lr_params = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    array_names = ['{}_{}'.format(prefix, n) for n in lr_params]

    # linregress faster without dask
    ds.load()
    # return a tuple of DataArrays
    res = xr.apply_ufunc(scipy.stats.linregress, ds[x], ds[y],
            input_core_dims=[dims, dims],
            output_core_dims=[[] for i in lr_params],
            vectorize=True,
            # dask='allowed',
            output_dtypes=[DTYPE for i in lr_params]
            )
    # add the data to the existing dataset
    for arr_name, arr in zip(array_names, res):
        ds[arr_name] = arr


def double_log(arr):
    """Return a linearized Gumbel distribution
    """
    return (np.log(np.log(1/arr))).astype(DTYPE)


def step21_ranking(annual_maxs):
    """Rank the annual maxs in time, in ascending order
    NOTE: Loaiciga & Leipnik (1999) call for the ranking to be done in desc. order,
    but this results in negative scale parameter, and wrong fitting

    return a Dataset
    """
    asc_rank = annual_maxs.load().rank(dim='year')
    # make sure that the resulting array is in the same order as the original
    ranks = asc_rank.rename('rank').astype(DTYPE).transpose(*annual_maxs.dims)
    # Merge arrays
    return xr.merge([annual_maxs, ranks])


def step22_gumbel_fit_loaiciga1999(ds):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    n_obs = ds.annual_max.count(dim='year')
    # Estimate probability F{x} with the Weibull formula
    ds['estim_prob'] = (ds['rank'] / (n_obs+1)).astype(DTYPE)
    # linearize
    ds['estim_prob_linear'] = double_log(ds['estim_prob'])
    # First fit
    linregress(ds, 'annual_max', 'estim_prob_linear', 'estim_prob_lr', ['year'])
    # get provisional gumbel parameters
    ds['loc_prov'] = -ds['estim_prob_lr_intercept']/ds['estim_prob_lr_slope']
    ds['scale_prov'] = -1/ds['estim_prob_lr_slope']
    # Analytic probability F(x) from Gumbel CDF
    z = (ds['annual_max'] - ds['loc_prov']) / ds['scale_prov']
    ds['analytic_prob'] = np.e**(-np.e**-z)
    # Get the final location and scale parameters
    ds['analytic_prob_linear'] = double_log(ds['analytic_prob'])
    linregress(ds, 'annual_max', 'analytic_prob_linear', 'analytic_prob_lr', ['year'])
    ds['loc_loaiciga'] = -ds['analytic_prob_lr_intercept']/ds['analytic_prob_lr_slope']
    ds['scale_loaiciga'] = -1/ds['analytic_prob_lr_slope']
    return ds


def step2bis_gumbel_fit_moments(ds):
    """Fit Gumbel using the method of moments
    (Maidment 1993, cited by Bougadis & Adamowki 2006)
    """
    magic_number1 = 0.45
    magic_number2 = 0.7797
    mean = ds['annual_max'].mean(dim='year')
    std = ds['annual_max'].std(dim='year')
    ds['loc_moments'] = mean - (magic_number1 * std)
    ds['scale_moments'] = magic_number2 * std
    return ds

#
# def step25_KS_test(ds):
#     """Perform the Kolmogorov–Smirnov test
#     """
#     print(scipy.stats.kstest(ds_sel['annual_max'].values, 'gumbel_r', (ds_sel['loc_final'], ds_sel['scale_final'])))
#     print(scipy.stats.kstest(ds_sel['annual_max'].values, 'gumbel_r', (loc_naive, scale_naive)))


def step3_duration_gradient(ds):
    """Take a Dataset as input containing the fitted gumbel parameters
    Fit a linear regression on the log of the parameters and the log of the duration
    Keep the regression parameters as variables
    """
    # compute the log
    var_list = ['duration', 'loc_loaiciga', 'scale_loaiciga']
    logvar_list = ['log_duration', 'log_location', 'log_scale']
    for var, log_var in zip(var_list, logvar_list):
        ds[log_var] = np.log10(ds[var])
    # Do the linear regression
    linregress(ds, 'log_duration', 'log_location', 'loc_lr', ['duration'])
    linregress(ds, 'log_duration', 'log_scale', 'scale_lr', ['duration'])


def pearsonr(x, y):
    """wrapper for the pearson r computation from scipy
    return only the r value
    """
    return scipy.stats.pearsonr(x, y)[0]


def step4_pearsonr(ds):
    """Compare the scaling gradients using the pearson r
    """
    x = ds['duration']**ds['loc_lr_slope']
    y = ds['duration']**ds['scale_lr_slope']
    res = xr.apply_ufunc(pearsonr, x, y,
            input_core_dims=[['duration'], ['duration']],
            # output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[DTYPE]
            )
    ds['scaling_pearsonr'] = res

def set_attrs(ds):
    """Set atrributes of a dataset
    """
    # set the originating_centre as an atribute, not a coordinate
    ds.attrs['originating_centre'] = str(ds.coords['originating_centre'].values)
    ds = ds.drop('originating_centre')
    return ds


def benchmark(ds):
    """Run the gumbel fit for a number of extract sizes
    print result to stdout
    """
    duration_list = []
    sizes = [(i+1)*5 for i in range(5)]
    sizes_sq = [i*i for i in sizes]
    for degrees in sizes:
        locator = dict(latitude=slice(degrees, 0),  # Latitudes are in descending order
                       longitude=slice(0, degrees))
        sel = ds.loc[locator]
        start = datetime.now()
        step2_gumbel_fit(sel)
        duration = datetime.now() - start
        duration_list.append(duration)
    dur_sec = [d.total_seconds() for d in duration_list]
    print({k:v for k, v in zip(sizes_sq, dur_sec)})


def amax_rob():
    """read a list of CSV.
    return a dataArray of annual maxima
    """
    arr_list = []
    for fpath, sname in [(KISUMU_AMS, 'kisumu'), (KAMPALA_AMS, 'kampala')]:
        df = pd.read_csv(fpath, index_col='year')
        # print(df)
        ds = df.to_xarray()
        annual_maxs = []
        for var_name in ds.data_vars:
            annual_max = ds[var_name]
            annual_max.name = 'annual_max'
            da = annual_max.expand_dims('duration')
            da.coords['duration'] = [int(var_name)*24]
            annual_maxs.append(da)
        da_site = xr.concat(annual_maxs, 'duration')
        da_site = da_site.expand_dims('site')
        da_site.coords['site'] = [sname]
        arr_list.append(da_site)
    return xr.concat(arr_list, 'site')


def logger(fields):
    log_file_path = os.path.join(DATA_DIR, LOG_FILENAME)
    with open(log_file_path, mode='a') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(fields)


def main():
    # Log file
    # logger(['operation', 'timestamp', 'cumul_sec'])

    with ProgressBar():
        start_time = datetime.now()
        # Load hourly data #
        # logger(['start computing annual maxima', str(start_time), 0])
        # hourly_path = os.path.join(DATA_DIR, HOURLY_FILE)
        # hourly = xr.open_zarr(hourly_path).chunk(HOURLY_CHUNKS)
        # hourly_kamp = hourly.sel(latitude=KAMPALA_COORD[0],
        #                          longitude=KAMPALA_COORD[1],
        #                          method='nearest')
        # print(hourly_kamp.max().compute())
        # hourly_extract = hourly.loc[EXTRACT]
        # print(hourly)

        # Get annual maxima #
        # annual_maxs = step1_annual_maxs_of_roll_mean(hourly, DURATIONS, TEMP_RES).chunk(ANNUAL_CHUNKS)
        # amax_path = os.path.join(DATA_DIR, ANNUAL_FILE)
        # annual_maxs.to_dataset().to_zarr(amax_path, mode='w', encoding=ANNUAL_ENCODING)

        # logger(['start ranking annual maxima', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # annual_maxs = xr.open_zarr(amax_path)['annual_max']
        # annual_maxs = amax_rob()  # to compare with Rob's values

        # Do the ranking
        # ds_ranked = step21_ranking(annual_maxs).chunk(ANNUAL_CHUNKS)
        # print(ds_ranked)
        # logger(['start writing ranks', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # encoding = copy.deepcopy(ANNUAL_ENCODING)
        # encoding['rank'] = GEN_FLOAT_ENCODING
        # rank_path = os.path.join(DATA_DIR, ANNUAL_FILE_RANK)
        # ds_ranked.to_zarr(rank_path, mode='w', encoding=encoding)


        # fit Gumbel #
        # logger(['start iterative gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # ds_ranked = set_attrs(xr.open_zarr(rank_path))#.loc[EXTRACT]
        # ds_fitted = step22_gumbel_fit_loaiciga1999(ds_ranked)
        gumbel_path = os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL)
        ds_fitted = xr.open_zarr(gumbel_path)
        # logger(['start moments gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        ds_fitted = step2bis_gumbel_fit_moments(ds_fitted.chunk(ANNUAL_CHUNKS))
        print(ds_fitted)
        # logger(['start writting results of gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        gumbel_path = os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL)
        encoding = {v:GEN_FLOAT_ENCODING for v in ds_fitted.data_vars.keys()}
        # ds_fitted.to_zarr(gumbel_path, mode='w', encoding=encoding)
        # print(ds_fitted)

        # fit duration scaling #
        # logger(['start duration scaling fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        step3_duration_gradient(ds_fitted)
        # logger(['start writing duration scaling', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # gradient_path = os.path.join(DATA_DIR, ANNUAL_FILE_GRADIENT)
        # encoding = {v:GEN_FLOAT_ENCODING for v in ds_fitted.data_vars.keys()}
        # ds_fitted.chunk(ANNUAL_CHUNKS).to_zarr(gradient_path, mode='w', encoding=encoding)

        # ds_fitted = xr.open_zarr(gradient_path)
        # logger(["start pearson's r computation", str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        step4_pearsonr(ds_fitted)
        # logger(["start writing pearson's r", str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        pearson_path = os.path.join(DATA_DIR, ANNUAL_FILE_PEARSONR)
        encoding = {v:GEN_FLOAT_ENCODING for v in ds_fitted.data_vars.keys()}
        ds_fitted.chunk(ANNUAL_CHUNKS).to_zarr(pearson_path, mode='w', encoding=encoding)
        print(ds_fitted)
        # logger(['complete', str(datetime.now()), (datetime.now()-start_time).total_seconds()])

        # display params to compare with Rob's
        # print(ds_fitted[['loc_final', 'scale_final']].loc[{'site':'kampala'}].to_dataframe().transpose())


if __name__ == "__main__":
    sys.exit(main())
