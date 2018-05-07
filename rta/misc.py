import numpy as np

def max_space(x):
    """Calculate the maximal space between a set of 1D points.

    If there is only one point, return 0.
    """
    if len(x) == 1:
        return 0
    return np.diff(np.sort(x)).max()
