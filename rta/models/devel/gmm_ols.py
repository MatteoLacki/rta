"""Check the distribution of the error terms from one run of Huber regression.

This way we will ascertain, if only one step of Huber fitting is necessary.
Alternatively, we could refit on the 'signal' data with either L2 or L1 regressors.
Possibly checking for the values of some parameters.
"""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices, dmatrix, bs, cr, cc
from sklearn.linear_model import HuberRegressor
from statsmodels import robust

from rta.models.base_model import cv, coef, predict
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation import grouped_K_folds, filter_foldable
from rta.models.GMM_OLS_combo import GMM_OLS
from rta.models.plot import plot

# get data
annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
K = 5 # number of folds
annotated_cv = filter_foldable(annotated, annotated_stats, K)
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds,
                                                  K = K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)
annotated_cv_1 = annotated_cv[annotated_cv.run == 1]
data = annotated_cv_1
data = data.sort_values(['rt', 'rt_median_distance'])

chunks_no = 100
gmm_ols = GMM_OLS()
formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no)

# An alternative for the calling of the function: directly pass X, y
# how will patsy handle directly passed a tupple?
x = gmm_ols.control
y = gmm_ols.response

o = y, x = dmatrices( (y, x) )


# np.percentile(xm q = )
from rta.array_operations.misc import percentiles_of_N_integers as percentiles

# need to pass them later on.
control_percentiles = np.fromiter((x[i] for i in percentiles(len(y), chunks_no)
                                   if i != chunks_no),
                                  dtype=np.float64,
                                  count=chunks_no - 1)

len(control_percentiles)

b_splines = bs(x,
               df=None,
               knots=control_percentiles,
               degree=0,
               include_intercept=False,
               lower_bound=0,
               upper_bound=200)

len(x[x <= control_percentiles[1]])
np.all(b_splines[:,0] == 0)

b_splines = bs(x,
               df=None,
               knots=np.percentile(x, q=[33,66]),
               degree=0,
               include_intercept=True,
               lower_bound=0,
               upper_bound=200)

b_splines

b_splines.shape


from rta.models.spline_regression import SplineRegression

class GMM_OLS(SplineRegression):
    def fit(self,
            formula,
            data={},
            data_sorted=False,
            chunks_no=100,
            **kwds):

































# sd = np.sqrt(gmm_ols.covariances[:,0])
# plt.plot(range(len(sd)), sd)
# plt.show()


# comparing the two fits
plt.subplot(2, 1, 1)
gmm_ols = GMM_OLS()
gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no)
plot(gmm_ols)
plt.ylim(-3,3) 

plt.subplot(2, 1, 2)
gmm_ols = GMM_OLS()
gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no, weighted=True)
plot(gmm_ols)

plt.ylim(-3,3) 
plt.show()


# Now: we have to sync the generation of spline with the percentiles of the control variable
# 
# should the formula be split?
# we need to call bs function directly because of the need to change df?
# maybe as a transform method???

# how patsy things are generated in the first place?
from patsy import bs, cc, cr


plt.title("B-spline basis example (degree=3)");
x = np.linspace(0., 1., 100)
y1= dmatrix(bs(x, df=6, degree=3, include_intercept=True))
y = dmatrix("bs(x, df=6, degree=1, include_intercept=True) - 1", {"x": x})



b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])
plt.plot(x, y*b);
plt.plot(x, np.dot(y, b), color='white', linewidth=3);
plt.show()

(y*b).shape

y1.design_info
y.design_info



control = gmm_ols.control

control[list(percentiles(len(control), chunks_no))]
list(percentiles(6, 3))



