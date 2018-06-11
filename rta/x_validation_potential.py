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

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.sklearn_regressors import SklearnRegression
from rta.xvalidation.grouped_k_folds import grouped_K_folds
from rta.xvalidation.filters import filter_foldable

path = "~/Projects/retentiontimealignment/Data/"
annotated, unlabelled = big_data(path=path)
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
# annotated_slim = annotated[['run', 'id',
#                                 'rt', 'rt_median_distance', 
#                                 'dt', 'dt_median_distance']]
# unlabelled_slim = unlabelled[['run', 
#                                 'rt', 'dt']]
# run CV     
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

for _, d in annotated_cv.groupby("run"):
    pass

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
y, X = map(np.asarray, dmatrices(formula, d))



def test():
    h_reg = HuberRegressor(warm_start = True,
                           fit_intercept = True,
                           alpha=.001)
    out = h_reg.fit(X, y.ravel())
    ps = PredefinedSplit(d.fold)

    cv_scores = cross_val_score(estimator = h_reg, 
                                X = X, 
                                y = y.ravel(),
                                cv = ps)
    return cv_scores



ps.get_n_splits()

print(ps)       

for train_index, test_index in ps.split():



# diff approach: neglect the precise composition; focus: number of runs
print("Sizes of runs_no-strata:\n", count(annotated_stats.runs_no))



