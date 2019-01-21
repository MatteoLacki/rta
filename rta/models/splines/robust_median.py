"""The Robust Spline class 2: the final countdown.

Here I try to use the robust median.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as lsq_spline
from scipy.interpolate import UnivariateSpline as u_spline
from scipy.interpolate import InterpolatedUnivariateSpline as i_spline

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline

from rta.models.splines.gaussian_mixture import gaussian_mixture_spline
from rta.align.calibrator       import calibrate
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess, filter_unfoldable

from masstodon.plot.spectrum import plot_spectrum
from collections import Counter
from rta.stats.stats import mad


folds_no    = 10
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
x_min = min(c.D.rt_0)
x_max = max(c.D.rt_0)

x = np.array(c.D.rt_0[c.D.run == 1])
y = np.array(c.D.runs_stat_dist_0[c.D.run == 1])

# how many things end up in 4 second intervals
Counter(Counter(np.round(x*15)).values())

# plt.scatter(x,y)
# plt.show()
# how much the RTs differ?
W = c.D.pivot(index='id', columns='run', values='rt_0')
W.columns = ["run_"+str(i+1) for i in range(10)]
W.run_1
plt.scatter(W.run_1,W.run_2)
plt.plot([x_min, x_max],[x_min, x_max], color='black')
plt.show()


from rta.array_operations.misc import percentile_pairs_of_N_integers

k_tile = 100
%%timeit
x_medians = np.empty(k_tile+4, dtype = np.float64)
y_medians = np.empty(k_tile+4, dtype = np.float64)
y_sds     = np.empty(k_tile+4, dtype = np.float64)
x_medians[0] = x_min - (x_max-x_min)/200.0
x_medians[1] = x_min
y_medians[0] = y_medians[1] = y_sds[0] = 0.0
i = 2
for s,e in percentile_pairs_of_N_integers(len(x), k_tile):
    x_medians[i] = x[(s+e)//2]
    y_mad, y_medians[i] = mad(y[s:e], return_median=True)
    y_sds[i] = 1.4826 * y_mad
    i += 1
x_medians[i] = x_max
x_medians[i+1] = x_max + (x_max-x_min)/200.0
y_medians[i] = y_medians[i+1] = y_sds[i+1] = 0.0
y_sds[1] = y_sds[2]/2.0
y_sds[i] = y_sds[i-1]/2.0

# scipy.interpolate.UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=100)
lsq_spline(x_medians, y_medians, np.percentile(x, np.linspace(0,100,20)))

median_interpolation = i_spline(x_medians, y_medians, k=3)
t_sd_interpolation = i_spline(x_medians, y_medians + 3*y_sds, k=3)
b_sd_interpolation = i_spline(x_medians, y_medians - 3*y_sds, k=3)

xs = np.linspace(x_min, x_max, 1000)
plt.scatter(x,y)
plt.plot(xs, median_interpolation(xs), color='black')
plt.plot(xs, t_sd_interpolation(xs), color='orange')
plt.plot(xs, b_sd_interpolation(xs), color='orange')
plt.show()


# seems that the approach with seconds should be better.
def get_conditional_means(x, y,
                           = 60,
                          sort = True):
    """Approximate the condifitional means by a spline through a moving median."""
     x, y = sort_by_x(x, y) if sort else (x, y)
     medians = np.empty()


class MedianSpline(Spline):
    def __init__(self,
                 )
