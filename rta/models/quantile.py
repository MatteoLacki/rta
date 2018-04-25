from patsy import dmatrices
import statsmodels.api as sm
from rta.models.spline_regression import SplineRegression


class QuantileSplineRegression(SplineRegression):
    """Quantile regression splines."""
    def fit(self, formula, q=.5, **kwds):
        if 'q' not in kwds:
            kwds['q'] = q
        self.dmatrices(formula)
        quantile_spline = sm.QuantReg(self.y, self.X)
        self.fit_out = quantile_spline.fit(**kwds)
        self.coef = self.fit_out.params

    def __repr__(self):
        return "This is a quantile spline regression."


def quantile_spline(data, formula, **kwds):
    rspline = QuantileSplineRegression(data)
    rspline.fit(formula, **kwds)
    return rspline
