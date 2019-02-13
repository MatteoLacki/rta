import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

from rta.align.calibrator       import calibrate
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.models.splines.gaussian_mixture import gaussian_mixture_spline
from rta.models.splines.beta_splines import beta_spline
from rta.array_operations.misc import dedup_sort


folds_no = 10
min_runs_no = 5
annotated_all, unlabelled_all = big_data()
d = preprocess(annotated_all, min_runs_no)
d = filter_unfoldable(d, folds_no)
c = calibrate(feature     ='rt',
              data        = d,
              folds_no    = folds_no,
              min_runs_no = min_runs_no,
              model       = gaussian_mixture_spline,
              align_cnt   = 0)

x_min = min(c.D.rt_0); x_max = max(c.D.rt_0)


x = np.array(c.D.rt_0[c.D.run == 1])
# y = np.array(c.D.runs_stat_dist_0[c.D.run == 1])
y = np.array(c.D.runs_stat_0[c.D.run == 1])

# x, y = dedup_sort(x,y)

# xs = np.linspace(x_min, x_max, 1000)
# rm_y = medfilt(y, 21)

# x[np.insert(np.diff(x,1) == 0, 0, False)]
# x[np.insert(np.diff(x,1) == 0, -1, False)]

# plt.scatter(x,y)
# plt.plot(x, rm_y, color='black')
# plt.plot(x, medfilt(y, 101), color='orange')
# plt.plot(x, medfilt(y, 151), color='red')
# plt.plot(x[::10], medfilt(y, 11)[::10], color='yellow')

# X = x[::10]
# Y = medfilt(y, 101)[::10]
# us = interp1d(X, Y,
#               bounds_error=False,
#               fill_value=0)
# xs = np.linspace(min(x), max(x), 1000)

# diff_quot = np.diff(Y)/np.diff(X)
# np.all(np.abs(diff_quot) < 1)
# sum(np.abs(diff_quot) >= 1) # we can live with that!

# plt.scatter(X[np.argwhere(np.abs(diff_quot) >= 1)],
#             Y[np.argwhere(np.abs(diff_quot) >= 1)])

# plt.plot(xs, us(xs), color='green')
# plt.show()

# is this of any good? apply the alignment and see!
# but this has to be applied to all the bloody runs
# ax = x - us(x)
# from rta.models.splines.spline import Spline

# class RollingMedianInterpolation(Spline):
#     def __init__(self,
#                  window_size=101,
#                  tf=10):
#         """Constructor.

#         Args:
#             window_size (int): an odd integer: the size of the window.
#             tf (int): each tf-th rolled median will be used for interpolation.
#         """
#         self.window_size = int(window_size) # should be odd too

#     def copy(self):
#         """Copy constructor."""
#         return RollingMedianInterpolation(self.window_size)

#     def fit(self, x, y, 
#             drop_duplicates = True,
#             sort            = True):
#         """Fit a rolilng median interpolator.

#         Args:
#             x (np.array):              The control variable.
#             y (np.array):              The response variable.
#             drop_duplicates (logical): Drop rows in xy dataframe where 'x' is not uniquely determined.
#             sort (logical):            Sort rows in xy dataframe w.r.t. 'x'.
#         """
#         self.set_xy(x, y, drop_duplicates, sort)
#         self.medians = medfilt(y, self.window_size)
#         self.spline = inter1d(self.x[::self.tf],
#                               self.medians[::self.tf])

#     def __repr__(self):
#         return "RolllingMedianInterpolation"

# rmi = RollingMedianInterpolation()
# rmi.plot()

#TODO y is always the difference to the star-RT
def rolling_median_interpolator(x, y,
        ws=101,
        k=10,
        sort=False):
    """Easier implementation, no OOP.

    Args:
        x (np.array): x values.
        y (np.array): y values.
        ws (odd int): Window size.
        k (int): Each k-th x entry will be used to make interpolation.
        sort (bool): Should we sort (x,y) by x?
    """
    if sort:
        i = np.argsort(x)
        x, y = x[i], y[i]
    medians = medfilt(y, ws)
    interpo = interp1d(x[::k], medians[::k],
                       bounds_error=False,
                       fill_value=0)
    return interpo

RMI = rolling_median_interpolator
rmi = RMI(x, y-x)
plt.scatter(x, y-x-rmi(x), s=1)
plt.show()


# the full model should be on the bloody runs!
from multiprocessing import Pool, cpu_count
import pandas as pd


class Model(object):
    def __init__(self, M, *M_args, **M_kwds):
        """Constuctor.
        Args:
            M (callable): the type of model to be fitted to model y-x.
        """
        self.M = M
        self.M_args = M_args
        self.M_kwds = M_kwds
        self.m = {}

    def fit(self, X):
        """Fit all coordinate models of distances to reference.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns r, x, y,
            where r - runs, x - observed, y - reference.
        """
        for r, Xr in X.groupby('r'):
            # y consists of reference RT
            self.m[r] = self.M(Xr.x.values,
                               Xr.y.values - Xr.x.values,
                               *self.M_args,
                               **self.M_kwds)
        # hence: we model the distance to the reference retention time.

    # to do: this might be not ordered according to the r values...
    def __call__(self, X):
        """Align the observations in X.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns r, x, where r - runs, and x - values to be aligned.
            ATTENTION! sort X by r!!!
        """
        dx = np.zeros(X.shape[0])
        i = 0
        for r, Xr in X.groupby('r'):
            n = Xr.shape[0]
            dx[i:i+n] = self.m[r](Xr.x.values)
            i += n
        return X.x.values + dx

    #TODO: this should return error per r-basis?
    def error(self, X):
        """Evaluate the error on X.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns r, x, y,
            where r - runs, x - data to be aligned, y - reference.
        """
        return np.median(np.abs(self(X) - X.y.values))

    def cv(self, X):
        """Cross-validate the model.

        Args:
            X (pd.DataFrame): dataframe with (at least) columns f, r, x, y, where f are folds, r - runs.
        Return:
            Average median test error across runs.
        """
        tot_err = 0
        f_vals = np.unique(X.f.values)
        for f in f_vals:
            X_train = X[X.f != f]
            X_test = X[X.f == f]
            m = Model(self.M, *self.M_args, **self.M_kwds)
            m.fit(X_train)
            tot_err += m.error(X_test)
        return tot_err / len(f_vals)


# TODO : add multiprocessing.

# some method should prepare X
X = c.D.copy()
X = X[['id', 'run', 'rt_0', 'runs_stat_0', 'fold']]
X.columns = ['p', 'r', 'x', 'y', 'f']
X.sort_values(by='r', inplace=True)
X = X.reset_index(drop=True)

M = Model(RMI, ws=31)
M.fit(X)
# M.error(X)
M.cv(X)

X2 = X.query('abs(x-y)<1')
plt.scatter(X.x, X.y - X.x, s=2, color='blue')
plt.scatter(X.x, X.y - M(X), s=1, color='orange')

# plt.scatter(X2.x, X2.y - M(X2), s=1)
plt.show()


plt.hist(X.x - X.y, color = 'blue', edgecolor = 'black',
         bins = 100)
plt.show()

# wss = np.arange(1, 100, 10) * 10 + 1
# wss = [11, 51, 101, 301]
wss = np.arange(3, 13, 2)
Models = [Model(RMI, ws=ws) for ws in wss]

def cv(M):
    return M.cv(X)
with Pool(4) as p:
    errs = p.map(cv, Models)

BM = Models[np.argmin(errs)]
BM.M_kwds

plt.plot(wss, errs)
plt.show()


BM = Model(RMI, ws=3)
x = X[X.r == 1].x.values
y = X[X.r == 1].y.values


xs = np.linspace(min(x), max(x), 1000)
plt.scatter(x, y)
plt.plot(xs, BM.m[1](xs))
plt.show()


# shit, this is not soo fast like the previous one.
# but might be more meaningful.




# def prepare_jobs(
#         IN,
#         M=rolling_median_interpolator,
#         **M_kwds):
#     for r, d_r in IN.groupby('r'):
#         x = d_r.rt_0.values
#         y = d_r.runs_stat_0.values
#         m = M(x, y, **M_kwds)
#         ax = x - m(x)
