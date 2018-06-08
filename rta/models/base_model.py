class Model(object):
    """A container for storing results of fitting."""
    def fit(self, formula, data={}, **kwds):
        raise NotImplementedError

    def fit_simple(self):
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
