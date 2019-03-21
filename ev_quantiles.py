# -*- coding: utf8 -*-

import sys
import math

import xarray as xr
import numpy as np
# from scipy.stats import gamma
from math import gamma

import helper

"""Collection of functions related to extreme values quantiles
"""

DTYPE = 'float32'
fscalar = np.float32
iscalar = np.int32


def lnt(T):
    return -np.log(1 - 1/T)

def y_gumbel(T):
    """
    Lu, L.-H., & Stedinger, J. R. (1992).
    Variance of two- and three-parameter GEV/PWM quantile estimators:
    formulae, confidence intervals, and a comparison.
    Journal of Hydrology, 138(1–2), 247–267.
    http://doi.org/10.1016/0022-1694(92)90167-T
    """
    return -np.log(lnt(T))

def y_gev_nonzero(T, shape):
    """
    Lu, L.-H., & Stedinger, J. R. (1992).
    Variance of two- and three-parameter GEV/PWM quantile estimators:
    formulae, confidence intervals, and a comparison.
    Journal of Hydrology, 138(1–2), 247–267.
    http://doi.org/10.1016/0022-1694(92)90167-T
    """
    return (1 - lnt(T)**shape) / shape


def gumbel_quantile(T, loc, scale):
    """Return quantile (i.e, intensity) for a given return period T in years
    """
    # y_gumbel() returns a negative
    return loc + scale * y_gumbel(T)


def gev_quantile_nonzero(T, loc, scale, shape):
    """Return quantile (i.e, intensity) for a given return period T in years
    Consider an EV type II if shape<0
    """
    return loc + scale * y_gev_nonzero(T, shape)


def gev_quantile(T, loc, scale, shape):
    """T: return period in years
    """
    return xr.where(shape == 0,
                    gumbel_quantile(T, loc, scale),
                    gev_quantile_nonzero(T, loc, scale, shape))


"""Following functions estimate of the quantile variance as in:
Lu, L.-H., & Stedinger, J. R. (1992).
Variance of two- and three-parameter GEV/PWM quantile estimators:
formulae, confidence intervals, and a comparison.
Journal of Hydrology, 138(1–2), 247–267.
http://doi.org/10.1016/0022-1694(92)90167-T
"""


def f_dloc_dbeta0(shape):
    term1 = 1 / (1 - 2**-shape)
    term2 = 1 / ((1 - 2**-shape) * gamma(1 + shape))
    return 1 - term1 + term2


def f_dloc_dbeta1(shape):
    term1 = 2 / (1 - 2**-shape)
    term2 = 2 / ((1 - 2**-shape) * gamma(1 + shape))
    return term1 - term2


def f_dscale_dbeta0(shape):
    return -shape / ((1 - 2**-shape) * gamma(1 + shape))


def f_dscale_dbeta1(shape):
    return 2*shape / ((1 - 2**-shape) * gamma(1 + shape))


def f_var_b0(scale, shape, n_obs):
    term1 = scale**2 / (n_obs * shape**2)
    term2 = gamma(1 + 2*shape) - gamma(1 + shape)**2
    return term1 * term2


def f_var_b1(scale, shape, n_obs):
    term1 = (2**(-2*shape) * scale**2) / (n_obs * shape**2)
    term2 = gamma(1 + 2 * shape) * f_G(.5, shape) - gamma(1 + shape)**2
    return term1 * term2


def f_cov_b0_b1(scale, shape, n_obs):
    term1 = scale**2 / (2 * n_obs * shape**2)
    term2 = 2**(-2*shape) * gamma(1 + 2*shape) + (1 - 2**(1-shape)) * gamma(1 + shape)**2
    return term1 * term2


def f_G_pwr(x, shape, m):
    num = gamma(2*shape + m) * (-x)**m
    denum = (shape + m) * math.factorial(m)
    return num / denum


def f_G(x, shape):
    arr_sum = f_G_pwr(x, shape, 1)
    for m in range(2, 30):
        arr_sum += f_G_pwr(x, shape, m)
    return 1 + (2*shape**2 / gamma(1 + 2*shape)) * arr_sum


def f_var_loc(scale, shape, n_obs):
    dloc_dbeta0 = f_dloc_dbeta0(shape)
    dloc_dbeta1 = f_dloc_dbeta1(shape)
    term1 = dloc_dbeta0 ** 2 * f_var_b0(scale, shape, n_obs)
    term2 = dloc_dbeta1 ** 2 * f_var_b1(scale, shape, n_obs)
    term3 = 2 * dloc_dbeta0 * dloc_dbeta1 * f_cov_b0_b1(scale, shape, n_obs)
    return term1 + term2 + term3


def f_var_scale(scale, shape, n_obs):
    dscale_dbeta0 = f_dscale_dbeta0(shape)
    dscale_dbeta1 = f_dscale_dbeta1(shape)
    term1 = dscale_dbeta0 ** 2 * f_var_b0(scale, shape, n_obs)
    term2 = dscale_dbeta1 ** 2 * f_var_b1(scale, shape, n_obs)
    term3 = 2 * dscale_dbeta0 * dscale_dbeta1 * f_cov_b0_b1(scale, shape, n_obs)
    return term1 + term2 + term3


def f_cov_loc_scale(scale, shape, n_obs):
    dloc_dbeta0 = f_dloc_dbeta0(shape)
    dloc_dbeta1 = f_dloc_dbeta1(shape)
    dscale_dbeta0 = f_dscale_dbeta0(shape)
    dscale_dbeta1 = f_dscale_dbeta1(shape)
    cov_b0_b1 = f_cov_b0_b1(scale, shape, n_obs)
    term1 = dloc_dbeta0 * dscale_dbeta0 * f_var_b0(scale, shape, n_obs)
    term2 = dloc_dbeta1 * dscale_dbeta1 * f_var_b1(scale, shape, n_obs)
    term3 = dloc_dbeta0 * dscale_dbeta1 * cov_b0_b1
    term4 = dloc_dbeta1 * dscale_dbeta0 * cov_b0_b1
    return term1 + term2 + term3 + term4


def f_c1(scale, shape, n_obs):
    var_loc = f_var_loc(scale, shape, n_obs)
    return n_obs * var_loc / scale**2


def f_c2(scale, shape, n_obs):
    cov_loc_scale = f_cov_loc_scale(scale, shape, n_obs)
    return 2 * n_obs * cov_loc_scale / scale**2


def f_c3(scale, shape, n_obs):
    var_scale = f_var_scale(scale, shape, n_obs)
    return n_obs * var_scale / scale**2


def f_c1_estim(shape):
    return 1.1128 - 0.2384*shape + 0.0908*shape**2 + 0.1084*shape**3


def f_c2_estim(shape):
    return 0.4580 - 3.0561*shape + 1.1104*shape**2 - 0.4071*shape**3


def f_c3_estim(shape):
    return 0.8046 - 2.8890*shape + 8.7874*shape**2 - 10.375*shape**3


def gev_quantile_var_fixed_shape(T, c1, c2, c3, da_gev, n_obs):
    """Return the variance of the GEV quantile for a given return period.
    See:
    Lu, L.-H., & Stedinger, J. R. (1992).
    Variance of two- and three-parameter GEV/PWM quantile estimators:
    formulae, confidence intervals, and a comparison.
    Journal of Hydrology, 138(1–2), 247–267.
    http://doi.org/10.1016/0022-1694(92)90167-T
    """
    shape = da_gev.sel(ev_param='shape', ci='estimate')
    scale = da_gev.sel(ev_param='scale', ci='estimate')
    y = xr.where(shape == 0,
                 y_gumbel(T),
                 y_gev_nonzero(T, shape))
    return scale*scale * (c1 + c2*y + c3*y*y) / n_obs


def print_c(scale, shape, n_obs):
    print('c1', f_c1(scale, shape, n_obs), f_c1_estim(shape))
    print('c2', f_c2(scale, shape, n_obs), f_c2_estim(shape))
    print('c3', f_c3(scale, shape, n_obs), f_c3_estim(shape))


if __name__ == "__main__":
    scale = 1
    shape = -0.114
    n_obs = 40
    sys.exit(print_c(scale, shape, n_obs))
