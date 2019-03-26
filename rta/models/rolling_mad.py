import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from rta.models.interpolant import Interpolant
from rta.models.spline import Spline
from rta.array_operations.dedupy import dedup_np
from rta.math.splines import beta as beta_spline

from pathlib import Path
data = Path("~/Projects/rta/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")

x = D.loc[D.run==1, 'rta'].values
r = D.loc[D.run==1, 'rta_med'].values
y = r-x

# import matplotlib.pyplot as plt
# plt.scatter(x, y, s=1)
# plt.show()



class RollingMAD(Interpolant):
    """The rolling Median Absolute Distance interpolator.

    Idea is as straight as a hair of a Mongol.
    """
    def __init__(self, ws=51, k=10, const=1.4826):
        """Constructor.
        
        Args:
            ws (odd int): window size.
            k (int): each k-th median will be used for interpolation
        """
        self.ws = ws
        self.k = k
        self.params = {'ws':ws, 'k':k}

    def __repr__(self):
        return "RollingMAD(ws:{} k:{})".format(self.ws, self.k)

    def fit(self, x, y, sort=True):
        """Fit the model.

        Args:
            x (np.array): The control variable.
            y (np.array): The response variable.
        """
        if sort:
            i = np.argsort(x)
            x, y = x[i], y[i]
        self.medians = medfilt(y, self.ws)
        self.mads = medfilt(np.abs(y - self.medians))
        self.x = x
        self.y = y
        self.interpo = interp1d(x[::self.k],
                                self.mads[::self.k],
                                bounds_error=False,
                                fill_value=0)

rMAD = RollingMAD()
rMAD.fit(x, y)
rMAD.plot()

len(y)
len(medfilt(y))


