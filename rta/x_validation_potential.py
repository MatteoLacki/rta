%load_ext autoreload
%autoreload 2
%load_ext line_profiler

# Possible optimizations:
    # make sequence & modification & run a multi-index.
    # aggregate accordingly w.r.t. the other dimensions.
    # assign dtypes other than "python objects"? 
from collections import Counter as count
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.model_selection import cross_val_score, PredefinedSplit

from rta.models.base_model import cv, coef
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.sklearn_regressors import SklearnRegression, sklearn_spline
from rta.models.huber import huber_spline
from rta.xvalidation.grouped_k_folds import grouped_K_folds
from rta.xvalidation.filters import filter_foldable

annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
# run CC
    # Runs seperately every of the 2**10 categories in 5 folds.
    # this way every run will have similar representation of points!
    # disregard points that cannot be cast into folds.
         # what the CV does with them anyway?
    # then, divide each fold into seperate runs and feed the models

K = 5 # number of folds
annotated_cv = filter_foldable(annotated, annotated_stats, K)
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds, K=K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)
# # sizes of folds withing groups should be similar
# for _, d in annotated_cv.groupby("run"):
#     a = count(d.fold)
#     print(sorted(list(a.items())))

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_validate
from patsy import dmatrices


def fit_huber(data, formula, cv=False):
    y, X = map(np.asarray, dmatrices(formula, data))    
    h_reg = HuberRegressor(warm_start = True,
                           fit_intercept = True,
                           alpha=.001,
                           epsilon=2.0)
    h_reg.fit(X, y.ravel())
    
    if cv:
        ps = PredefinedSplit(data.fold)
        cv_scores = cross_val_score(estimator = h_reg, 
                                    X = X, 
                                    y = y.ravel(),
                                    cv = ps)
    else:
        cv_scores = None
    ## this works well on Linux, good!
    #                             n_jobs=-1,
    #                             pre_dispatch='16')
    return h_reg, cv_scores

# what do we need now?
# run the code for every run.

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

for _, data in annotated_cv.groupby('run'):
    pass

h_reg, cv_scores = fit_huber(data, formula, cv=False)
h_reg.intercept_
h_reg.scale_
h_reg.sigma_


plt.scatter(data.rt,
            data.rt_median_distance,
            c = h_reg.outliers_)
plt.show()
# wait, shouldn't we implement it in the general code?


from sklearn.utils.estimator_checks import check_estimator

class NaiveRegressor():
    def fit(self, X, y):
        return np.full(y.shape, 0)

    def score(self):
        return 0
check_estimator(NaiveRegressor)
naive_regressor = NaiveRegressor()

cross_val_score(estimator = naive_regressor, 
                X = data.rt,
                y = data.rt_median_distance)


# Extending the class to match sklearn

from sklearn.base import BaseEstimator, RegressorMixin


h_spline = huber_spline(formula=formula,
                        data=d,
                        warm_start=True)
cv(h_spline)
coef(h_spline)
# TODO 
    # why the R**2 is so low? It seems impossible!
    # learn how to define your own metrics.
    # calculate some average Huber error (make it not increase with obs no)
    # leave the strategy of using the Huber regressor alone for now.
    # read how to do grid search on the df in the splines arg.



# How can this be done efficiently?




# diff approach: neglect the precise composition; focus: number of runs
print("Sizes of runs_no-strata:\n", count(annotated_stats.runs_no))



# annotated_slim = annotated[['run', 'id',
#                                 'rt', 'rt_median_distance', 
#                                 'dt', 'dt_median_distance']]
# unlabelled_slim = unlabelled[['run', 
#                                 'rt', 'dt']]
