# -*- coding: utf8 -*-

import xarray as xr
import statsmodels.api as sm

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


def OLS(da_x, da_y, dim):
    """Linear regression along the dimension dim.
    Use Ordinary Least Squares.
    """
    mean_x = da_x.mean(dim=dim)
    mean_y = da_y.mean(dim=dim)
    x_diff = da_x - mean_x
    slope = ((x_diff * (da_y - mean_y)).sum(dim=dim) /
             (x_diff**2).sum(dim=dim)).rename('line_slope')
    intercept = (mean_y - slope * mean_x).rename('line_intercept')

    # coefficient of determination
    fitted = slope * da_x + intercept
    rsquared = ((np.square(fitted - mean_y).sum(dim=dim)) /
        (np.square(da_y - mean_y).sum(dim=dim))).rename('rsquared')

    return slope, intercept, rsquared
