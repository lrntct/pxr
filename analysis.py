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
import scipy.optimize
import statsmodels.api as sm


DATA_DIR = '/home/lunet/gylc4/geodata/ERA5/'
# DATA_DIR = '../data/MIDAS/'
HOURLY_FILE1 = 'midas_2000-2017_precip_pairs.nc'
# HOURLY_FILE2 = 'era5_2013-2017_precip.zarr'

# ANNUAL_FILE = 'midas_2000-2017_precip_annual_max.nc'
# ANNUAL_FILE2 = 'era5_2013-2017_precip_annual_max.zarr'
# ANNUAL_FILE_GUMBEL = 'era5_2000-2017_precip_gumbel.zarr'
# ANNUAL_FILE_GRADIENT = 'era5_2000-2017_precip_gradient.zarr'
ANNUAL_FILE_SCALING = 'midas_2000-2017_precip_pairs_scaling.nc'

LOG_FILENAME = 'Analysis_log_{}.csv'.format(str(datetime.now()))

# Extract
# EXTRACT = dict(latitude=slice(1.0, -0.25),
#                longitude=slice(32.5, 35))
EXTRACT = dict(latitude=slice(0, -5),
               longitude=slice(0, 5))

# Event durations in hours - has to be adjusted to temporal resolution for the moving window
# Selected to be equally spaced on a log scale. Manually adjusted from a call to np.geomspace()
DURATIONS_SUBDAILY = [1, 2, 3, 4, 6, 8, 10, 12, 18, 24]
DURATIONS_DAILY = [24, 48, 72, 96, 120, 144, 192, 240, 288, 360]
# use fromkeys to remove duplicate. need py >= 3.6 to preserve order
DURATIONS_ALL = list(dict.fromkeys(DURATIONS_SUBDAILY + DURATIONS_DAILY))
DURATION_DICT = {'all': DURATIONS_ALL, 'daily': DURATIONS_DAILY, 'subdaily': DURATIONS_SUBDAILY}
# DURATION_DICT = {'daily': DURATIONS_DAILY}
# Temporal resolution of the input in hours
TEMP_RES = 1
# TEMP_RES = 24


LR_RES = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
LOGISTIC_PARAMS = ['max', 'steepness', 'midpoint']

DTYPE = 'float32'

HOURLY_CHUNKS = {'time': -1, 'latitude': 16, 'longitude': 16}
ANNUAL_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 45*4, 'longitude': 45*4}  # 4 cells: 1 degree
EXTRACT_CHUNKS = {'year': -1, 'duration':-1, 'latitude': 4, 'longitude': 4}
GAUGES_CHUNKS = {'year': -1, 'duration':-1, 'station': 200}
GEN_FLOAT_ENCODING = {'dtype': DTYPE, 'compressor': zarr.Blosc(cname='lz4', clevel=9)}
ANNUAL_ENCODING = {'annual_max': GEN_FLOAT_ENCODING,
                   'duration': {'dtype': DTYPE},
                   'latitude': {'dtype': DTYPE},
                   'longitude': {'dtype': DTYPE}}


def linregress(func, x, y, dims):
    """x, y: dataArray to use for the regression
    dims: dimension on which to carry out the regression
    """
    # return a tuple of DataArrays
    res = xr.apply_ufunc(func, x, y,
            input_core_dims=[dims, dims],
            output_core_dims=[[] for i in LR_RES],
            vectorize=True,
            dask='allowed',
            output_dtypes=[DTYPE for i in LR_RES]
            )
    return res
    # add the data to the existing dataset
    # for arr_name, arr in zip(array_names, res):
    #     ds[arr_name] = arr


def nanlinregress(x, y):
    """wrapper around statsmodels OLS to make it behave like scipy linregress.
    Make use of its capacity to ignore NaN.
    """
    X = sm.add_constant(x)
    try:
        results = sm.OLS(y, X, missing='drop').fit()
    except ValueError:
        slope = np.nan
        intercept = np.nan
        rvalue = np.nan
        pvalue = np.nan
        stderr = np.nan
    else:
        slope = results.params[1]
        intercept = results.params[0]
        rvalue = results.rsquared ** .5
        pvalue = results.pvalues[1]
        stderr = results.bse[1]

    return slope, intercept, rvalue, pvalue, stderr


def rank_annual_maxs(annual_maxs):
    """Rank the annual maxs in time, in ascending order
    NOTE: Loaiciga & Leipnik (1999) call for the ranking to be done in desc. order,
    but this results in negative scale parameter, and wrong fitting.
    return a Dataset
    """
    asc_rank = annual_maxs.load().rank(dim='year')
    # make sure that the resulting array is in the same order as the original
    ranks = asc_rank.rename('rank').astype(DTYPE).transpose(*annual_maxs.dims)
    # Merge arrays in a single dataset
    return xr.merge([annual_maxs, ranks])


def gumbel_cdf(x, loc, scale):
    z = (x - loc) / scale
    return np.e**(-np.e**-z)


def gumbel_scipy_stats_fit(y):
    try:
        params = scipy.stats.gumbel_r.fit(y)
    except RuntimeError:
        params = (np.nan, np.nan)
    return params[0], params[1]


def gumbel_scipy_fit(ds):
    """Employ scipy stats to find the Gumbel coefficients
    """
    loc, scale = xr.apply_ufunc(gumbel_scipy_stats_fit,
                                ds['annual_max'],
                                input_core_dims=[['year']],
                                output_core_dims=[[], []],
                                vectorize=True,
                                dask='allowed',
                                output_dtypes=[DTYPE, DTYPE]
                                )
    return loc.rename('location'), scale.rename('scale')


def gumbel_fit_loaiciga1999(ds):
    """Follow the steps described in:
    Loaiciga, H. A., & Leipnik, R. B. (1999).
    Analysis of extreme hydrologic events with Gumbel distributions: marginal and additive cases.
    Stochastic Environmental Research and Risk Assessment (SERRA), 13(4), 251–259.
    https://doi.org/10.1007/s004770050042
    """
    # linearize
    linearize = lambda a: (np.log(np.log(1/a))).astype(DTYPE)
    estim_prob_linear = linearize(ds['estim_prob'])
    # First fit. Keep only the two first returning DataArrays
    estim_slope, estim_intercept = linregress(scipy.stats.linregress,
                                              ds['annual_max'],
                                              estim_prob_linear, ['year'])[:2]
    # get provisional gumbel parameters
    loc_prov = -estim_intercept / estim_slope
    scale_prov = -1 / estim_slope
    # Analytic probability F(x) from Gumbel CDF
    analytic_prob = gumbel_cdf(ds['annual_max'], loc_prov, scale_prov)
    # Get the final location and scale parameters
    analytic_prob_linear = linearize(analytic_prob)
    analytic_slope, analytic_intercept = linregress(scipy.stats.linregress,
                                                    ds['annual_max'],
                                                    analytic_prob_linear, ['year'])[:2]
    loc_final = (-analytic_intercept / analytic_slope).rename('location')
    scale_final = (-1 / analytic_slope).rename('scale')
    return loc_final, scale_final


def gumbel_fit_moments(ds):
    """Fit Gumbel using the method of moments
    (Maidment 1993, cited by Bougadis & Adamowki 2006)
    """
    magic_number1 = 0.45
    magic_number2 = 0.7797
    mean = ds['annual_max'].mean(dim='year')
    std = ds['annual_max'].std(dim='year')
    loc = (mean - (magic_number1 * std)).rename('location')
    scale = (magic_number2 * std).rename('scale')
    return loc, scale


def KS_test(ds):
    """Perform the Kolmogorov–Smirnov test on the Gumbel fitting
    """
    ds = ds.chunk({'duration':-1})
    ds['analytic_prob'] = gumbel_cdf(ds['annual_max'], ds['location'], ds['scale'])
    ds['ks'] = xr.apply_ufunc(lambda x,y: np.max(np.abs(x-y)),
                                    ds['estim_prob'], ds['analytic_prob'],
                                    input_core_dims=[['year'], ['year']],
                                    vectorize=True,
                                    dask='parallelized',
                                    output_dtypes=[DTYPE]
                                    )
    # Dcrit at the 0.05 confidence
    len_dur = len(ds['duration'])
    ds['Dcrit_5pct'] = 1.36 * np.sqrt((len_dur+len_dur)/(len_dur*len_dur))
    return ds


def fit_logistic(x, y):
    """Fit a logitic funtion to the data using scipy
    """
    # remove nan
    finite = np.logical_and(np.isfinite(x), np.isfinite(y))
    nanx = x[finite]
    nany = y[finite]
    # print(nanx)
    # print(nany)
    def logistic_func(x, mv, s, mp): return mv / (1 + np.exp(-s*(x-mp)))
    try:
        popt, pcov = scipy.optimize.curve_fit(logistic_func, nanx, nany, p0=[-1, 1, 2], maxfev=1000)
    # RuntimeError: no optimal parameters are found; TypeError: not enough datapoints
    except (RuntimeError, TypeError):
        print("warning: failed logistic regression for y: {}".format(nany))
        popt = [np.nan for i in range(3)]
    # print(popt)
    return popt[0], popt[1], popt[2]


def logistic_regression(x, y, dims):
    # return a tuple of DataArrays
    res = xr.apply_ufunc(fit_logistic, x, y,
                        input_core_dims=[dims, dims],
                        output_core_dims=[[] for i in LOGISTIC_PARAMS],
                        vectorize=True,
                        dask='allowed',
                        output_dtypes=[DTYPE for i in LOGISTIC_PARAMS]
                        )
    # add the data to the existing dataset
    # for arr_name, arr in zip(array_names, res):
    #     ds[arr_name] = arr
    return res


def pearsonr(x, y):
    """wrapper for the pearson r computation from scipy
    """
    return scipy.stats.pearsonr(x, y)


def spearmanr(x, y):
    """wrapper for the Spearman's r computation from scipy
    return only the r value
    """
    try:
        return scipy.stats.spearmanr(x, y, nan_policy='omit')[0]
    except ValueError:
        return np.nan


def anderson_gumbel(x):
    try:
        return scipy.stats.anderson(x, dist='gumbel_r')[0]
    except RuntimeError:
        return np.nan


def anderson_darling(ds):
    """
    """
    # Get the critical values (depend only on the sample length)
    _, critical_values, significance_levels = scipy.stats.anderson(ds['year'], dist='gumbel_r')
    da_critical_values = xr.DataArray(critical_values, name='A2_crit',
                                      coords=[significance_levels],
                                      dims=['significance_level'])
    # Goodness of fit of the Gumbel distribution
    da_a2 = xr.apply_ufunc(anderson_gumbel,
                           ds['annual_max'],
                           input_core_dims=[['year']],
                           vectorize=True,
                           dask='parallelized',
                           output_dtypes=[DTYPE]
                           ).rename('A2')
    return da_critical_values, da_a2


def ttest(x, y):
    """wrapper for the Student's t test for independent samples from scipy
    return t-statistic and pvalue
    """
    return scipy.stats.ttest_ind(x, y, axis=0, nan_policy='omit')


def step1_annual_maxs_of_roll_mean(ds, precip_var, time_dim, durations, temp_res):
    """for each rolling winfows size:
    compute the annual maximum of a moving mean
    return an array with the durations as a new dimension
    """
    annual_maxs = []
    for duration in durations:
        window_size = int(duration / temp_res)
        precip = ds[precip_var]
        precip_roll_mean = precip.rolling(**{time_dim:window_size}, min_periods=max(int(window_size*.9), 1)).mean(dim=time_dim, skipna=True)
        annual_max = precip_roll_mean.groupby('{}.year'.format(time_dim)).max(dim=time_dim, skipna=True)
        annual_max.name = 'annual_max'
        da = annual_max.expand_dims('duration')
        da.coords['duration'] = [duration]
        annual_maxs.append(da)
    return xr.concat(annual_maxs, 'duration')


def step2_fit_gumbel(da_annual_maxs):
    """Fit gumbel using various techniques.
    Keep them along a new dimension
    """
    n_obs = da_annual_maxs.count(dim='year')
    # Add rank and turn to dataset
    ds = rank_annual_maxs(da_annual_maxs)
    # Estimate probability F{x} with the Weibull formula
    ds['estim_prob'] = (ds['rank'] / (n_obs+1)).astype(DTYPE)
    # Goodness of fit of the Gumbel distribution
    anderson_darling(ds)
    # Do the fitting
    ds_list = []
    for fit_func, name in [(gumbel_fit_moments, 'moments'),
                           #(gumbel_fit_loaiciga1999, 'iterative_linear'),
                           (gumbel_scipy_fit, 'scipy'),
                           ]:
        ds_fit = xr.merge(fit_func(ds))
        ds_fit = ds_fit.expand_dims('gumbel_fit')
        ds_fit.coords['gumbel_fit'] = [name]
        ds_list.append(ds_fit)
    ds_fit = xr.concat(ds_list, dim='gumbel_fit')
    ds = xr.merge([ds, ds_fit])
    # Perform the Kolmogorov-Smirnov test
    ds = KS_test(ds)
    return ds


def step3_scaling(ds):
    """Take a Dataset as input containing the fitted gumbel parameters
    Fit a linear and logistic functions on the log of the parameters and the log of the duration
    Keep the regression parameters as variables
    The fitting is done on various duration series kept as a new dimension
    """
    # log-transform the variables
    var_list = ['duration', 'location', 'scale']
    logvar_list = ['log_duration', 'log_location', 'log_scale']
    for var, log_var in zip(var_list, logvar_list):
        ds[log_var] = np.log10(ds[var])

    ds_list = []
    for dur_name, durations in DURATION_DICT.items():
        # Select only the durations of interest
        ds_sel = ds.sel(duration=durations)
        da_list = []
        for g_param_name in ['location', 'scale']:
            param_col = 'log_{}'.format(g_param_name)
            # Do the linear regression. Keep only the 3 first results
            slope, intercept, rvalue = linregress(nanlinregress,
                                                  ds_sel['log_duration'],
                                                  ds_sel[param_col],
                                                  ['duration'])[:3]
            # Fit a logistic function
            maximum, steepness, midpoint = logistic_regression(ds_sel['log_duration'],
                                                               ds_sel[param_col],
                                                               ['duration'])

            for var_name, da in zip(['line_slope', 'line_intercept', 'line_rvalue',
                                     'logistic_max', 'logistic_steepness', 'logistic_midpoint'],
                                    [slope, intercept, rvalue, maximum, steepness, midpoint]):
                da.name = '{}_{}'.format(g_param_name, var_name)
                da_list.append(da)
        # Group all DataArrays in a single dataset
        ds_fit = xr.merge(da_list)
        # Keep the the results in their own dimension
        ds_fit = ds_fit.expand_dims('scaling_extent')
        ds_fit.coords['scaling_extent'] = [dur_name]
        ds_list.append(ds_fit)
    ds_fit = xr.concat(ds_list, dim='scaling_extent')
    # Add thos DataArray to the general Dataset
    ds = xr.merge([ds, ds_fit])
    return ds


def adhoc_rvalues(ds):
    """
    """
    ds_list = []
    for extent_name in ds['scaling_extent']:
        # Select only the durations of interest
        ext_name = extent_name.values.item().decode('utf-8')
        ds_sel = ds.sel(scaling_extent=extent_name, duration=DURATION_DICT[ext_name]).load()
        da_list = []
        for g_param_name in ['location', 'scale']:
            param_col = 'log_{}'.format(g_param_name)
            rvalue, pvalue = xr.apply_ufunc(pearsonr,
                                    ds_sel['log_duration'], ds_sel[param_col],
                                    input_core_dims=[['duration'], ['duration']],
                                    output_core_dims=[[], []],
                                    vectorize=True,
                                    # dask='parallelized',
                                    output_dtypes=[DTYPE, DTYPE]
                                    )
            rvalue.name = '{}_{}'.format(g_param_name, 'line_rvalue')
            pvalue.name = '{}_{}'.format(g_param_name, 'line_pvalue')
            da_list.extend([rvalue, pvalue])
        # Group all DataArrays in a single dataset
        ds_fit = xr.merge(da_list)
        # Keep the the results in their own dimension
        # ds_fit = ds_fit.expand_dims('scaling_extent')
        # ds_fit.coords['scaling_extent'] = [dur_name]
        ds_list.append(ds_fit)
    ds_fit = xr.concat(ds_list, dim='scaling_extent')
    # rename variables for consistency
    # ds = ds.rename({'location_slope': 'location_line_slope', 'location_intercept': 'location_line_intercept',
    #                 'location_max': 'location_logistic_max', 'location_midpoint': 'location_logistic_midpoint', 'location_steepness': 'location_logistic_steepness',
    #                 'scale_slope': 'scale_line_slope', 'scale_intercept': 'scale_line_intercept',
    #                 'scale_max': 'scale_logistic_max', 'scale_midpoint': 'scale_logistic_midpoint', 'scale_steepness': 'scale_logistic_steepness',})
    # Add those DataArray to the general Dataset
    return xr.merge([ds, ds_fit])


def adhoc_AD(annual_max):
    """Calculate Anderson-Darling statistic for each duration.
    Save a separate file for each duration
    """
    for duration in annual_max['duration']:
        duration_value = duration.values
        print(duration_value)
        da_sel = annual_max.sel(duration=duration)
        da_a2 = xr.apply_ufunc(anderson_gumbel,
                               da_sel.load(),
                               input_core_dims=[['year']],
                               vectorize=True,
                            #    dask='parallelized',
                               output_dtypes=[DTYPE]
                               ).rename('A2')
        out_name = 'era5_2000-2017_precip_a2_{}.nc'.format(duration_value)
        out_path = os.path.join(DATA_DIR, out_name)
        da_a2.to_dataset().to_netcdf(out_path)


def step4_scaling_correlation(ds):
    """Compare the scaling gradients using:
    Pearson's r
    Spearman's r
    Student's t
    ratio
    """
    ds_list = []
    for dur_name, durations in DURATION_DICT.items():
        # Select only the durations of interest
        ds_sel = ds.sel(duration=durations)
        scaling_pearsonr = xr.apply_ufunc(pearsonr,
                                        ds_sel['location'], ds_sel['scale'],
                                        input_core_dims=[['duration'], ['duration']],
                                        # output_core_dims=[[]],
                                        vectorize=True,
                                        dask='parallelized',
                                        output_dtypes=[DTYPE]
                                        ).rename('scaling_pearsonr')
        scaling_spearmanr = xr.apply_ufunc(spearmanr,
                                        ds_sel['location'], ds_sel['scale'],
                                        input_core_dims=[['duration'], ['duration']],
                                        # output_core_dims=[[]],
                                        vectorize=True,
                                        dask='parallelized',
                                        output_dtypes=[DTYPE]
                                        ).rename('scaling_spearmanr')
        ttest_res = xr.apply_ufunc(ttest,
                                ds_sel['location'], ds_sel['scale'],
                                input_core_dims=[['duration'], ['duration']],
                                output_core_dims=[[], []],
                                vectorize=True,
                                dask='allowed',
                                output_dtypes=[DTYPE, DTYPE]
                                )
        scaling_student_t = ttest_res[0].rename('scaling_student_t')
        scaling_student_pvalue = ttest_res[1].rename('scaling_student_pvalue')
        # Group all DataArrays in a single dataset
        ds_fit = xr.merge([scaling_pearsonr, scaling_spearmanr,
                           scaling_student_t, scaling_student_pvalue])
        # Keep the the results in their own dimension
        ds_fit = ds_fit.expand_dims('scaling_extent')
        ds_fit.coords['scaling_extent'] = [dur_name]
        ds_list.append(ds_fit)
    # Add the new Datasets to the general Dataset
    ds = xr.merge([ds, ds_fit])
    # In that case, the computation is automatically done per scaling extent
    ds['scaling_ratio'] = ds['location_line_slope'] / ds['scale_line_slope']
    return ds


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
        hourly_path1 = os.path.join(DATA_DIR, HOURLY_FILE1)
        # hourly_path2 = os.path.join(DATA_DIR, HOURLY_FILE2)
        hourly1 = xr.open_dataset(hourly_path1)#.chunk(HOURLY_CHUNKS).loc[EXTRACT]
        # hourly2 = xr.open_zarr(hourly_path2)
        # hourly = set_attrs(xr.concat([hourly1, hourly2], dim='time')).chunk(HOURLY_CHUNKS)#.loc[EXTRACT]
        # hourly = xr.open_zarr(hourly_path1)
        print(hourly1)

        # Get annual maxima #
        # annual_maxs = step1_annual_maxs_of_roll_mean(hourly1, 'prcp_amt', 'end_time', DURATIONS_ALL, TEMP_RES)#.chunk(ANNUAL_CHUNKS)
        # annual_maxs = step1_annual_maxs_of_roll_mean(hourly, 'precipitation', 'time', DURATIONS_DAILY, TEMP_RES)#.chunk(ANNUAL_CHUNKS)
        # amax_path = os.path.join(DATA_DIR, ANNUAL_FILE)
        # amax_path2 = os.path.join(DATA_DIR, ANNUAL_FILE2)
        # print(annual_maxs.load())
        # encoding = {v:GEN_FLOAT_ENCODING for v in annual_maxs.to_dataset().data_vars.keys()}
        # annual_maxs.to_dataset().to_zarr(amax_path, mode='w', encoding=encoding)
        # annual_maxs.to_dataset().to_netcdf(amax_path, mode='w')

        # fit Gumbel #
        # ds_fitted = step2_fit_gumbel(annual_maxs)#.chunk(ANNUAL_CHUNKS)
        # print(ds_fitted)
        # logger(['start writting results of gumbel fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # gumbel_path = os.path.join(DATA_DIR, ANNUAL_FILE_GUMBEL)
        # encoding = {v:GEN_FLOAT_ENCODING for v in ds_fitted.data_vars.keys()}
        # ds_fitted.to_zarr(gumbel_path, mode='w', encoding=encoding)
        # print(ds_fitted)

        # fit parameters scaling #
        # ds_fitted = xr.open_zarr(gumbel_path)#.loc[EXTRACT]
        # logger(['start duration scaling fitting', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # ds_scaled = step3_scaling(ds_fitted)
        # print(ds_scaled.load())
        # logger(['start writing duration scaling', str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # gradient_path = os.path.join(DATA_DIR, ANNUAL_FILE_GRADIENT)
        # encoding = {v:GEN_FLOAT_ENCODING for v in ds_scaled.data_vars.keys()}
        # ds_scaled.chunk(ANNUAL_CHUNKS).to_zarr(gradient_path, mode='w', encoding=encoding)

        # ds_fitted = xr.open_zarr(gradient_path)
        # logger(["start correlation computation", str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # ds_scaled = step4_scaling_correlation(ds_scaled)#.chunk(ANNUAL_CHUNKS)
        # print(ds_scaled.load())
        # print(ds_scaled['ks'].load().quantile([0.95,0.99,0.999], dim=['duration', 'latitude', 'longitude']))
        # print(ds_scaled['location'].load().std(dim=['duration', 'latitude', 'longitude']))

        # logger(["start writing correlation", str(datetime.now()), (datetime.now()-start_time).total_seconds()])
        # scaling_path = os.path.join(DATA_DIR, ANNUAL_FILE_SCALING)
        # encoding = {v:GEN_FLOAT_ENCODING for v in ds_scaled.data_vars.keys()}
        # ds_scaled.to_zarr(scaling_path, mode='w', encoding=encoding)
        # print(ds_fitted.load())
        # ds_scaled.to_netcdf(scaling_path)
        # print(ds_scaled.compute())
        # print((~np.isfinite(ds_fitted)).sum().compute())
        # logger(['complete', str(datetime.now()), (datetime.now()-start_time).total_seconds()])

        # path1 = os.path.join(DATA_DIR, 'era5_2000-2017_precip_scaling_renamed.zarr')
        # path2 = os.path.join(DATA_DIR, 'era5_2000-2017_precip_rvalue_pvalue.zarr')
        # ds1 = xr.open_zarr(path1)#.chunk(ANNUAL_CHUNKS)#.loc[EXTRACT]
        # ds2 = xr.open_zarr(path2)
        # adhoc_AD(ds1['annual_max'])
        # ds_combined = xr.merge([ds1, ds2]).chunk(ANNUAL_CHUNKS)
        # print(ds_combined)
        # print(ds_scaled)
        # ds_rvalues = adhoc_rvalues(ds1).chunk(ANNUAL_CHUNKS)#[['location_line_rvalue', 'scale_line_rvalue']]
        # print(ds_rvalues)
        # encoding = {v: GEN_FLOAT_ENCODING for v in ds_combined.data_vars.keys()}
        # ds_combined.to_zarr(os.path.join(DATA_DIR, 'era5_2000-2017_precip_renamed_rvalue.zarr'),
        #                    mode='w', encoding=encoding)
        # print(ds_rvalues['scaling_extent'])
        # print(ds_combined)
        # da_critical_values, da_a2 = anderson_darling(ds_scaled)
        # ds_anderson = xr.merge([da_critical_values, da_a2])
        # print(ds_anderson)
        # ds_anderson.to_netcdf(os.path.join(DATA_DIR, 'era5_2000-2017_precip_a2.nc'))
        # encoding = {v:GEN_FLOAT_ENCODING for v in ds_combined.data_vars.keys()}
        # ds_combined.to_zarr(os.path.join(DATA_DIR, 'era5_2000-2017_precip_renamed_rvalue.zarr'),
        #                     mode='w', encoding=encoding)


        # combine all A2 files
        a2_path = os.path.join(DATA_DIR, 'era5_2000-2017_precip_a2_*.nc')
        ds_a2 = xr.open_mfdataset(a2_path, concat_dim='duration')
        da_a2 = ds_a2.reindex(duration=sorted(ds_a2.duration))['A2']
        # Open 
        ds_era = xr.open_zarr(os.path.join(DATA_DIR, 'era5_2000-2017_precip_renamed_rvalue.zarr'))
        # Get the critical values
        _, crit_val, sig_levels = scipy.stats.anderson(ds_era['year'],
                                                       dist='gumbel_r')
        da_crit_val = xr.DataArray(crit_val, name='A2_crit',
                                   coords=[sig_levels],
                                   dims=['significance_level'])
        # Add those dataArray to the main dataset
        ds_era = xr.merge([ds_era, da_a2, da_crit_val]).chunk(ANNUAL_CHUNKS)
        print(ds_era)
        # Save to zarr
        encoding = {v: GEN_FLOAT_ENCODING for v in ds_era.data_vars.keys()}
        ds_era.to_zarr(os.path.join(DATA_DIR, 'era5_2000-2017_precip_complete.zarr'),
                            mode='w', encoding=encoding)


if __name__ == "__main__":
    sys.exit(main())
