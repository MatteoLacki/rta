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
from rta.array_operations.misc import percentiles

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

# chunks_no = 100
# gmm_ols = GMM_OLS()
# formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
# gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no)



# TODO:
    # reproduce the spline fitting with state-transforms
        # adjust the predict function
    # the fitting of splines can potentially be done repeatedly, so we will need these classes to 
    # accept new arguments. 
        # set warm starts everywhere
        # splines will also need to be recalculated.
# Question: how is beta spline evalution implemented by PATSY? Check source code.
    # it calls scipy's implementation of Bsplines, that is a wrapper around some Fortran code


from scipy.interpolate import LSQUnivariateSpline as Spline
from collections import Counter as count
from rta.models.base_model import Model

# making the data friendly for the Fortran Spline fitting.




# assume x has no repeats
class GMM_OLS(Model):
    def __init__(self):
        self.have_data = False

    def prepare_df(self, data, x_name='x', y_name='y'):
        data = data.drop_duplicates(subset=x_name, keep=False, inplace=False)
        data = data.sort_values([x_name, y_name])
        self.x, self.y = (np.asarray(data[name]) for name in (x_name, y_name))
        self.have_data = True

    def fit(self,
            x=None,
            y=None,
            chunks_no=100,
            **kwds):
        """Fit a denoised spline."""
        pass

    def predict(self):
        pass

    def fitted(self):
        pass

    @property
    def res(self):
        """Get residuals."""
        pass

    def __repr__(self):
        """Represent the model."""
        #TODO make this more elaborate.
        return "This is a GMM_OLS combo class for super-duper fitting."    


gmm_ols = GMM_OLS()
data_no_dups = data.drop_duplicates(subset='rt', keep=False, inplace=False)
gmm_ols.prepare_df(data, 'rt', 'rt_median_distance')

x, y = gmm_ols.x, gmm_ols.y
chunks_no = 100 

%%timeit
x_inner_percentiles = percentiles(x, chunks_no, inner=True)
spline = Spline(x, y, x_inner_percentiles)























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



