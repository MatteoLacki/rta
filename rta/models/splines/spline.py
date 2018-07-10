import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.models.base_model import Model
from rta.stats.stats import mad, mae, confusion_matrix


def dedup_sort(x, y):
    """Remove dulicate x entries in x and for the corresponding y indices. 
    Sort by x."""
    d = pd.DataFrame({'x':x, 'y':y})
    d = d.drop_duplicates(subset='x', keep=False)
    d = d.sort_values(['x'])
    return d.x.values, d.y.values


class Spline(Model):
    """Abstract class for splines."""

    def plot(self,
             knots_no = 1000,
             plt_style = 'dark_background',
             show = True, 
             **kwds):
        """Plot the spline."""
        plt.style.use(plt_style)
        plt.scatter(self.x, self.y)
        xs = np.linspace(min(self.x), max(self.x), knots_no)
        ys = self.spline(xs)
        plt.plot(xs, ys, c='orange')
        if show:
            plt.show()

    def set_xy(self, x, y, drop_duplicates_and_sort=True):
        assert len(x) == len(y)
        self.x, self.y = dedup_sort(x, y) if drop_duplicates_and_sort else (x, y)

    def res(self):
        """Get residuals."""
        return self.y - self.fitted()

    def coef(self):
        return self.spline.get_coeffs()

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is the Spline."

    def new(self):
        """Create an instance of the same class."""
        return self.__class__()

    def cv(self, folds,
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
        n = self.new()
        for fold in np.unique(folds):
            x_train = self.x[folds != fold]
            y_train = self.y[folds != fold]
            x_test  = self.x[folds == fold]
            y_test  = self.y[folds == fold]
            m_signal = self.signal[folds == fold]
            n.fit(x_train,
                  y_train,
                  self.chunks_no,
                  self.std_cnt,
                  drop_duplicates_and_sort=False)
            errors = np.abs(n.predict(x_test) - y_test)
            n_signal = n.is_signal(x_test, y_test)
            s = [stat(errors) for stat in fold_stats]
            S.append(s)
            cm = confusion_matrix(m_signal, n_signal)
            FS.append((n, s, cm))

        S = np.array(S)
        S = np.array([stat(S, axis=0) for stat in model_stats])
        S = pd.DataFrame(S)
        S.columns = ["fold_" + fs.__name__ for fs in fold_stats]
        S.index = [ms.__name__ for ms in model_stats]

        self.cv_stats = S
        self.fold_stats = FS
