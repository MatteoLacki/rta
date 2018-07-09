import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.models.base_model import Model


def drop_duplicates_and_sort(x, y):
    """Remove dupilcate x entries. Sort by x."""
    d = pd.DataFrame({'x':x, 'y':y})
    d = d.drop_duplicates(subset='x', keep=False)
    d = d.sort_values(['x'])
    return d.x.values, d.y.values


class Spline(Model):
    """Abstract class for splines."""

    def drop_duplicates_and_sort(self, x, y):
        """Drop duplicate entries of x in both x and y and sort by x."""
        self.x, self.y = drop_duplicates_and_sort(x, y)

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

    def res(self):
        """Get residuals."""
        return self.y - self.fitted()

    def coef(self):
        return self.spline.get_coeffs()

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is the Spline."
