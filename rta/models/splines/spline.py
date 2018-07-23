import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.array.operations   import dedup_sort
from rta.models.base_model  import Model
from rta.stats.stats        import mad, mae, confusion_matrix


def dedup_sort(x, y, 
               drop_duplicates=True,
               sort=True):
    """Remove dulicate x entries in x and for the corresponding y indices. 
    Sort by x."""
    if drop_duplicates or sort:
        d = pd.DataFrame({'x':x, 'y':y})
        if drop_duplicates:
            d = d.drop_duplicates(subset='x', keep=False)
        if sort:
            d = d.sort_values(['x'])
        return d.x.values, d.y.values
    else:
        return x, y


class Spline(Model):
    """Abstract class for splines."""

    def plot(self,
             knots_no = 1000,
             plt_style = 'dark_background',
             show = True):
        """Plot the spline.

        Noise points are blue, signal has the color of papaya.

        Args:
            knots_no (int):  number of points used to plot the fitted spline?
            plt_style (str): the matplotlib style used for plotting.
            show (logical):  show the plot immediately. Alternatively, add some more elements on the canvas before using it.
        """
        plt.style.use(plt_style)
        colors = np.full(self.signal.shape, "blue", dtype='<U30')
        colors[self.signal] = 'grey'
        plt.scatter(self.x, self.y, c=colors)
        xs = np.linspace(min(self.x), max(self.x), knots_no)
        ys = self.spline(xs)
        plt.plot(xs, ys, c='orangered', linewidth=4)
        if show:
            plt.show()

    def set_xy(self, x, y,
               drop_duplicates=True,
               sort=True):
        assert len(x) == len(y)
        self.x, self.y = dedup_sort(x, y, drop_duplicates, sort)

    def predict(self, x):
        return self.spline(x)

    def fitted(self):
        return self.spline(self.x)

    def res(self):
        """Get residuals."""
        return self.y - self.fitted()

    def coef(self):
        return self.spline.get_coeffs()

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is the Spline."

    def copy(self):
        """Copy constructor."""
        raise NotImplementedError("This has to be coded subclass-specifical: they do differ in parametrization.")

    def cv(self,
           folds,
           fold_stats = (mae, mad),
           model_stats= (np.mean, np.median, np.std),
           confusion  = True):
        """Run cross-validation.

        Results are saved as class fields.

        Args:
            folds (np.array) an array of ints marking fold assignment.
            fold_stats (iterable of callables) statistics to apply to the errors in each fold.
            model_stats (iterable of callables) statistics to apply to fold_stats
            confusion (boolean) calculate confusion matrix
        """
        assert len(self.x) == len(folds)
        S = []
        FS  = []
        n = self.copy()
        for fold in np.unique(folds):
            x_train = self.x[folds != fold]
            y_train = self.y[folds != fold]
            x_test  = self.x[folds == fold]
            y_test  = self.y[folds == fold]
            m_signal = self.signal[folds == fold]
            n.fit(x_train,
                  y_train,
                  drop_duplicates=False,
                  sort=False)
            errors = np.abs(n.predict(x_test) - y_test)
            n_signal = n.is_signal(x_test, y_test)
            s = [stat(errors) for stat in fold_stats]
            S.append(s)
            cm = confusion_matrix(m_signal, n_signal)
            FS.append((s, cm))

        S = np.array(S)
        S = np.array([stat(S, axis=0) for stat in model_stats])
        S = pd.DataFrame(S)
        S.columns = ["fold_" + fs.__name__ for fs in fold_stats]
        S.index = [ms.__name__ for ms in model_stats]

        self.cv_stats = S
        self.fold_stats = FS
