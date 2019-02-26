import numpy as np


def is_angry(x, std_no=5, no_0=True):
    """Find points that are more than MAD (angry).

    Estimate standard deviation by median absolute distance
    and cut away things farther than 'std_no' from median.

    Args:
        x (np.array): 1D floats
        std_no (float): number of
    Return:
        np.array of booleans: True if is angry, False otherwise.
    """
    y = x[x != 0] if no_0 else x
    y_me = np.median(y)
    y_mad = np.median(np.abs(y - y_me))
    # k is for Gaussian data
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    k = 1.4826
    return np.abs(x - y_me) > std_no*y_mad*k