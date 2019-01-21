class Model(object):
    """A container for storing results of fitting."""
    def fit(self, x, y, **kwds):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        raise NotImplementedError

    #TODO: is this really needed?
    def refit(self, **kwds):
        """Refit the model."""
        raise NotImplementedError

    def __call__(self, x, *args, **kwds):
        """Predict the values at the new data points.

        Args:
            x (np.array): The control variable.
            *args       : Additional arguments.
            **kwds      : Additional arguments.

        Returns:
            np.array : Predicted values corresponding to values of 'x'.

        """
        raise NotImplementedError

    #TODO: is this really needed?
    def predict(self, x, *args, **kwds):
        """Predict the values at the new data points.

        Args:
            x (np.array): control variable.
            *args       : additional arguments.
            **kwds      : additional arguments.
        """
        self(x, y, *args, **kwds)

    #TODO: is this really needed? Some methods don't have coefs.
    def coef(self):
        """Retrieve spline coefficient.

        Then again, why would you?
        """
        raise NotImplementedError

    def res(self):
        """Get residuals."""
        raise NotImplementedError

    def residuals(self):
        """Get residuals: syntactic sugar for 'res'."""
        return self.res

    def __repr__(self):
        return 'This is the logic for all models.'

    def plot(self, **kwds):
        """Plot results."""
        raise NotImplementedError

    def cv(self, x, y, folds):
        """Run cross-validation."""
        raise NotImplementedError


def predict(model, x, *args, **kwds):
    """Predict the values at the new data points.

    Args:
        model       : The model of interest.
        x (np.array): The control variable.
        *args       : Additional arguments.
        **kwds      : Additional arguments.

    Returns:
        np.array : Predicted values corresponding to values of 'x'.

    """
    return model(x, y, *args, **kwds)


def fitted(model):
    return model.fitted()

def coef(model):
    return model.coef()

def coefficients(model):
    return model.coef()

def residuals(model):
    return model.res()

def res(model):
    return model.res()

def cv(model, **kwds):
    return model.cv(**kwds)

def plot(model, **kwds):
    model.plot(**kwds)