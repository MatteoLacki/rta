%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt

from rta.models.base_model import predict, fitted, coef, residuals
from rta.models.plot import plot
from rta.models.least_squares_splines import least_squares_spline
from rta.models.huber import huber_spline
from rta.models.RANSAC import RANSAC_spline
from rta.models.theil_sen import theil_sen_spline
from rta.models.quantile import quantile_spline

from rta.read_in_data import DT as D
from rta.preprocessing import preprocess


DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
# Huber regression
hspline = huber_spline(data, formula)
%matplotlib
plot(hspline)
# coef(hspline)
# residuals(hspline)
x_pred_range = np.arange(0, 200, .5)


# Theil-Sen: extremely slow
ts_spline = theil_sen_spline(data, formula)
plt.plot(x_pred_range,
         predict(ts_spline, rt = x_pred_range),
         c = 'gold')


# RANSAC regression
# WARNING: have to watch for the terms here!
RANSACspline = RANSAC_spline(data, "rt_median_distance ~ bs(rt, df=40, degree=3, lower_bound=0, upper_bound=200)")

plt.plot(x_pred_range,
         predict(RANSACspline, rt = x_pred_range),
         c = 'green')


# Quantile regression: slowest
qspline = quantile_spline(data, formula)
plt.plot(x_pred_range,
         predict(qspline, rt = x_pred_range),
         c = 'blue')

qspline_10 = quantile_spline(data, formula, q=.1)
plt.plot(x_pred_range,
         predict(qspline_10, rt = x_pred_range),
         c='orange')

qspline_90 = quantile_spline(data, formula, q=.9)
plt.plot(x_pred_range,
         predict(qspline_90, rt = x_pred_range),
         c = 'orange')


# least squares spline regression
l2_spline = least_squares_spline(data, "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200)")
plt.plot(x_pred_range,
         predict(l2_spline, rt = x_pred_range),
         c = 'green')
