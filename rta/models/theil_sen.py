from rta.models.sklearn_regressors import sklearn_spline


def theil_sen_spline(data, formula, **kwds):
    """Fit a Theil-Sen spline regression model."""
    return sklearn_spline(data, formula, regressor='Theil-Sen', **kwds)
