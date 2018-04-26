from rta.models.sklearn_regressors import sklearn_spline


def huber_spline(data, formula, **kwds):
    """Fit a Huber spline regression model."""
    return sklearn_spline(data, formula, regressor='Huber', **kwds)
