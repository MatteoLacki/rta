try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


class Model(object):
    """A container for storing results of fitting."""
    def fit(self, x, y, **kwds):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
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

    def __repr__(self):
        return 'Model'

    def fitted(self):
        """Get the fitted values."""
        return self(self.x)

    def res(self):
        """Get model residuals."""
        return self.y - self.fitted()

    def plot(self, plt_style='dark_background', show=True, **kwds):
        """Plot results."""
        if plt:
            plt.style.use(plt_style)
            plt.scatter(self.x, self.y, **kwds)
            if show:
                plt.show()
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError


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

def residuals(model):
    return model.res()

def res(model):
    return model.res()

def plot(model, **kwds):
    model.plot(**kwds)