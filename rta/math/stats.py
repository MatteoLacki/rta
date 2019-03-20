import numpy as np


def centiles(x):
    """Get centiles of x"""
    return np.quantile(x, [i/100 for i in range(101)])


def med_mad(x, const=1.4826):
    """Calculate the median and the absolute deviation from median.
    
    Args:
        x (np.array): 1D floats
        const: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    x_med = np.median(x)
    x_mad = np.median(np.abs(x-x_med)) * 1.4826
    return x_med, x_mad


def robust_chebyshev_interval(x, std_cnt=5, const=1.4826):
    """Calculate a region ±std_cnt * std_est times from the median.

    std_est is calculated as MAD * const.

    Args:
        x (np.array): 1D floats
        std_cnt (positive float): how many std should be considered?
        const: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    med, mad = med_mad(x, const)
    return med - std_cnt * mad, med + std_cnt * mad