from patsy import dmatrices
from rta.models.spline_regression import SplineRegression
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
from sklearn.linear_model import RANSACRegressor


class SklearnRegression(SplineRegression):
    """Sklearn-based regression splines."""
    def fit(self, formula, data={}, regressor_name="Huber", **kwds):
        regressors = {'Theil-Sen': TheilSenRegressor,
                      'Huber': HuberRegressor,
                      'RANSAC': RANSACRegressor}
        assert regressor_name in regressors, "God gonna cut you down."
        self.y, self.X = dmatrices(formula, data)

        if regressor_name is not 'RANSAC':
            kwds['fit_intercept'] = not 'include_intercept=True' in self.X.design_info.term_names[0]
        self.regressor = regressors[regressor_name](**kwds)
        self.fit_out = self.regressor.fit(self.X, self.y.ravel())
        if regressor_name is 'RANSAC':
            self.coef = self.fit_out.estimator_.coef_
        else:
            self.coef = self.fit_out.coef_

    def __repr__(self):
        return "This is sklearn spline regression."

    def cv(self, folds=None):
        pass
        # if not folds:
        #     folds = 



def sklearn_spline(formula,
                   data={},
                   regressor_name="Huber",
                   **kwds):
    """Fit one of the sklear regression splines."""
    sklearn_spline = SklearnRegression()
    sklearn_spline.fit(formula, data, regressor_name)
    return sklearn_spline
