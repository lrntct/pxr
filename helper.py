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
