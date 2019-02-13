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

    def copy(self):
        """Copy a model (including its parameters)."""
        return 

    def __repr__(self):
        return 'Model'

    def fitted(self):
        """Get the fitted values."""
        return self(self.x)

    def res(self):
        """Get model residuals."""
        return self.y - self.fitted()

    def plot(self, plt_style='dark_background', show=True, **kwds):
        """Plot the model's results.

        Args:
            plt_style (str): The style of the matplotlib visualization. Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
            show (bool): Show the figure, or just add it to the canvas.
            kwds: optional keyword arguments for matplotlib.plt
        """
        if plt:
            plt.style.use(plt_style)
            plt.scatter(self.x, self.y, **kwds)
            if show:
                plt.show()
        else:
            print('Install matplotlib to use this function.')
            raise ModuleNotFoundError

    def plot_residuals(self, plt_style='dark_background', show=True, **kwds):
        """Plot the model's residuals.

        Args:
            plt_style (str): The style of the matplotlib visualization. Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
            show (bool): Show the figure, or just add it to the canvas.
            kwds: optional keyword arguments for matplotlib.plt
        """
        if plt:
            plt.style.use(plt_style) 
            plt.scatter(self.x, self.res(), **kwds)
            plt.plot((0,max(self.x)), (0,0), c='orange')
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
    """Plot the model's outcomes."""
    model.plot(**kwds)