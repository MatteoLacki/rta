import numpy as np
from patsy import dmatrices
from rta.models.spline_regression import SplineRegression
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score, PredefinedSplit


class SklearnRegression(SplineRegression):
    """Sklearn-based regression splines."""
    def fit(self,
            formula, 
            data={},
            regressor_name="Huber",
            cv = False,
            **kwds):
        """Fit the model."""
        regressors = {'Theil-Sen': TheilSenRegressor,
                      'Huber': HuberRegressor,
                      'RANSAC': RANSACRegressor}
        assert regressor_name in regressors, "God gonna cut you down."

        self.data = data
        self.y, self.X = dmatrices(formula, self.data)
        self.y = self.y.ravel()

        if regressor_name is not 'RANSAC':
            kwds['fit_intercept'] = not 'include_intercept=True' in self.X.design_info.term_names[0]
        self.regressor = regressors[regressor_name](**kwds)
        self.fit_out = self.regressor.fit(self.X, self.y)

        if regressor_name is 'RANSAC':
            self.coef = self.fit_out.estimator_.coef_
        else:
            self.coef = self.fit_out.coef_
        self.coef = np.asarray(self.coef, dtype=np.float64)

    def __repr__(self):
        return "This is sklearn spline regression."

    def cv(self, folds=None, **kwds):
        """Perform cross validation of the final model."""
        
        if folds:
            cv = PredefinedSplit(folds)
        else:
            try:
                cv = PredefinedSplit(self.data.fold)   
            except AttributeError:
                cv = None

        self.cv_scores = cross_val_score(estimator = self.regressor,
                                         X = self.X,
                                         y = self.y,
                                         cv = cv,
                                         **kwds)
        return self.cv_scores



def sklearn_spline(formula,
                   data={},
                   regressor_name="Huber",
                   **kwds):
    """Fit one of the sklear regression splines."""

    sklearn_spline = SklearnRegression()
    sklearn_spline.fit(formula, data, regressor_name, **kwds)

    return sklearn_spline
