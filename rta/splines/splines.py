%load_ext autoreload
%autoreload 2

from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import bs, cr, cc
import matplotlib.pyplot as plt

from rta.models.base_model import predict, fitted, coef, residuals
from rta.models.least_squares_splines import least_squares_spline
from rta.models.huber import huber_spline
from rta.models.quantile import quantile_spline
from rta.models.plot import plot

from rta.models.sklearn_regressors import sklearnRegression, sklearn_spline
from rta.models.RANSAC import RANSAC_spline
from rta.models.theil_sen import theil_sen_spline
from rta.models.huber import huber_spline

%matplotlib

D = preprocess(D)
for g, data in D.groupby('run'):
    pass
formula = "rt_median_distance ~ bs(rt, df=40, degree=2, include_intercept=True) - 1"


# sklearn regression: long and inaccurate
sk_spline = sklearn_spline(data, formula, 'Theil-Sen')
plot(sk_spline)
coef(sk_spline)

# Huber regression
hspline = huber_spline(data, formula)
plot(hspline)

# RANSAC regression
# WARNING: have to watch for the terms here!
RANSACspline = RANSAC_spline(data, "rt_median_distance ~ bs(rt, df=40, degree=2)")
plot(RANSACspline)


# Quantile regression
qspline = quantile_spline(data, formula)

x_range = plot(qspline,
               c='gold',
               out_x_range=True)

qspline_10 = quantile_spline(data, formula, q=.1)
y_range_10 = predict(qspline_10, rt = x_range)
plt.plot(x_range, y_range_10, c='green')

qspline_90 = quantile_spline(data, formula, q=.9)
y_range_90 = predict(qspline_90, rt = x_range)
plt.plot(x_range, y_range_90, c='blue')


# least squares spline regression
l2_spline = least_squares_spline(data, "rt_median_distance ~ bs(rt, df=40, degree=2)")
predict(l2_spline, rt=[134, 140, 180])
fitted(l2_spline)
coef(l2_spline)
plot(l2_spline)


# sr.least_squares()
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ bs(rt, df=40, degree=2, include_intercept=True) - 1")
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ bs(rt, df=100, degree=4, include_intercept=True) - 1")
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ cr(rt, df=20)")
# sr.plot()
#
