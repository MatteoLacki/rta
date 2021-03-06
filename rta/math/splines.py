from scipy.interpolate import LSQUnivariateSpline as spline

from rta.array_operations.misc import percentiles


def beta(x, y, k=20, **args):
    """Fit spline to denoised data with least squares.

    The inner knots are chosen to be the approximate percentiles.

    Args:
        x (np.array): Knots, increasing and with no duplicates.
        y (np.array): values to approximate.
        k (int): The number of percentiles used as nodes for the analysis.
        args: Other keyword arguments to the LSQUnivariateSpline function.
    """
    x_inner_percentiles = percentiles(x, k, inner=True)
    try:
        s = spline(x = x, 
                   y = y,
                   t = x_inner_percentiles,
                   **args)
    except ValueError as ve:
        raise ValueError('Too many chunks to fit to.\nKnots are placed between chunks.\nReduce the number of chunks or give me more points.')
    return s
