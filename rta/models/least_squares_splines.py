from patsy import dmatrices
import numpy as np

from rta.models.spline_regression import SplineRegression


class LeastSquareSpline(SplineRegression):
    """Least squares splines."""
    def fit(self, formula, **kwds):
        if 'rcond' not in kwds:
            kwds['rcond'] = None
        self.dmatrices(formula)
        self.fit_out = np.linalg.lstsq(a=self.X,
                                       b=self.y,
                                       **kwds)
        self.coef = self.fit_out[0].ravel()

    def __repr__(self):
        return "This is a least squares spline regression."


def least_squares_spline(data, formula):
    spline_l2 = LeastSquareSpline(data)
    spline_l2.fit(formula)
    return spline_l2
