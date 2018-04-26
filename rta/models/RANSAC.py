from rta.models.sklearn_regressors import sklearn_spline


def RANSAC_spline(data, formula, **kwds):
    """Fit a RANSAC spline regression model."""
    return sklearn_spline(data, formula, regressor='RANSAC', **kwds)
