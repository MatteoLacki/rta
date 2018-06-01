from patsy import dmatrices
from rta.models.spline_regression import SplineRegression
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
from sklearn.linear_model import RANSACRegressor


class sklearnRegression(SplineRegression):
    """Sklearn-based regression splines."""
    def fit(self, formula, regressor, **kwds):
        regressor_name = regressor
        regressors = {'Theil-Sen': TheilSenRegressor,
                      'Huber': HuberRegressor,
                      'RANSAC': RANSACRegressor}
        assert regressor in regressors, "God gonna cut you down."
        self.dmatrices(formula) # this has to be called so that ↓↓↓↓↓↓ works
        if regressor_name is not 'RANSAC':
            kwds['fit_intercept'] = not 'include_intercept=True' in self.X.design_info.term_names[0]
        regressor = regressors[regressor](**kwds)
        self.fit_out = regressor.fit(self.X, self.y.ravel())
        if regressor_name is 'RANSAC':
            self.coef = self.fit_out.estimator_.coef_
        else:
            self.coef = self.fit_out.coef_

    def __repr__(self):
        return "This is sklearn spline regression."


def sklearn_spline(data, formula, regressor, **kwds):
    """Fit one of the sklear regression splines."""
    sklearn_spline = sklearnRegression(data)
    sklearn_spline.fit(formula, regressor)
    return sklearn_spline
