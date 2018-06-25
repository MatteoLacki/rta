import numpy as np


def mae(x):
    """Calculate the mean absolute distance."""
    return np.mean(np.abs(x))


def mad(x, return_median=False):
    """Compute median absolute deviation from median."""
    median = np.median(x)
    if return_median:
        return np.median(np.abs(x - median)), median
    else:
        return np.median(np.abs(x - median))


def confusion_matrix(real, pred):
    """A confusion matrix that is quicker than SKLearn one."""
    return np.array([[np.sum(np.logical_and( real, pred)), np.sum(np.logical_and(~real, pred))],
                     [np.sum(np.logical_and( real,~pred)), np.sum(np.logical_and(~real,~pred))]], 
                     dtype=int)
