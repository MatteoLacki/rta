from patsy import dmatrices
from sklearn.linear_model import HuberRegressor
from rta.models.spline_regression import SplineRegression


class HuberSplineRegression(SplineRegression):
    """Huber regression splines."""
    def fit(self, formula, **kwds):
        self.dmatrices(formula)
        kwds['fit_intercept'] = not 'include_intercept=True' in self.X.design_info.term_names[0]
        huber = HuberRegressor(**kwds)
        self.fit_out = huber.fit(self.X, self.y.ravel())
        self.coef = self.fit_out.coef_

    def __repr__(self):
        return "This is a quantile spline regression."


def huber_spline(data, formula, **kwds):
    """Fit a Huber spline regression model."""
    hspline = HuberSplineRegression(data)
    hspline.fit(formula)
    return hspline
