from patsy import dmatrices
import numpy as np

from rta.models.spline_regression import SplineRegression


class LeastSquareSpline(SplineRegression):
    """Least squares splines."""
    def fit(self, formula):
        self.dmatrices(formula)
        b, self.res, self.pred_rank, self.svals = np.linalg.lstsq(self.X,
                                                                  self.y,
                                                                  None)
        self.coef = b.ravel()

    def __repr__(self):
        return "This is a least squares spline regression."


def least_squares_spline(data, formula):
    spline_l2 = LeastSquareSpline(data)
    spline_l2.fit(formula)
    return spline_l2
