import numpy as np

from rta.math.stats import med_mad

def is_angry(x, std_no=5, no_0=False):
    """Find points that are more than MAD (angry).

    Estimate standard deviation by median absolute distance
    and cut away things farther than 'std_no' from median.

    Args:
        x (np.array): 1D floats
        std_no (float): number of standard deviations.
    Return:
        np.array of booleans: True if is angry, False otherwise.
    """
    y = x[x != 0] if no_0 else x
    y_med, y_mad = med_mad(y)
    return np.abs(x - y_med) > std_no * y_mad