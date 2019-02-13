import numpy as np


def centiles(x):
    """Get centiles of x"""
    return np.quantile(x, [i/100 for i in range(101)])
