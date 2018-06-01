class Model(object):
    """The logic of a model."""
    def __init__(self, data):
        self.data = data

    def fit(self, formula, **kwds):
        raise NotImplementedError

    def predict(self, newdata={}, *args, **kwds):
        """Predict the values at the new data points."""
        raise NotImplementedError

    @property
    def coefficients(self):
        return self.coef

    @property
    def res(self):
        """Get residuals."""
        raise NotImplementedError

    def residuals(self):
        """Get residuals: syntactic sugar for 'res'."""
        return self.res

    def __repr__(self):
        return 'This is the logic for all models.'

    def plot(self):
        raise NotImplementedError

    def cv(self):
        """Run cross-validation."""
        pass
        # for p in self.get_params_for_cv():
        #     for f in self.get_fold():

    def get_params_for_cv(self):
        """Run parameters for cross-validation."""
        raise NotImplementedError

    def get_folds_for_cv(self):
        """Run parameters for cross-validation."""
        raise NotImplementedError


def predict(model, newdata={}, *args, **kwds):
    return model.predict(newdata, *args, **kwds)

def fitted(model):
    return model.fitted()

def coef(model):
    return model.coef

def coefficients(model):
    return model.coefficients

def residuals(model):
    return model.residuals()

def res(model):
    return model.res
