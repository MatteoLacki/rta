from rta.models.sklearn_regressors import sklearn_spline


def huber_spline(formula, data, **kwds):
    """Fit a Huber spline regression model."""
    return sklearn_spline(formula=formula, 
    					  data=data, 
    					  regressor_name='Huber', 
    					  **kwds)
