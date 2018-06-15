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

from rta.models.base_model import coef, predict, fitted, coefficients, residuals, res, cv
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation import grouped_K_folds, filter_foldable
from rta.models.GMLSQSpline import GMLSQSpline, fit_spline, fit_interlapping_mixtures
from rta.models.plot import plot



annotated, unlabelled = big_data() # get data
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
K = 5 # number of folds
annotated_cv = filter_foldable(annotated, annotated_stats, K)
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds, K = K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)
annotated_cv_1 = annotated_cv[annotated_cv.run == 1]
data = annotated_cv_1
data = data.sort_values(['rt', 'rt_median_distance'])

# %%timeit
model = GMLSQSpline()
model.df_2_data(data, 'rt', 'rt_median_distance')
chunks_no = 10
model.fit(chunks_no = chunks_no, warm_start=True)

# plot(model)
# plt.ylim(-3,3) 
# plt.show()

coef(model)
predict(model, [10, 1223.232])
fitted(model)
coefficients(model)
residuals(model)
res(model)



# investigating different strategies of fitting to make it faster: premature optimization at its high!
# I woke up this morning, and deep deep, I couldn't sleep. I found out, I was to become the Jesus of 
# Saint Premature Optimization, walking the path of shame...
from rta.models.RapidGMLSQSpline import RapidGMLSQSpline 


rapid_model = RapidGMLSQSpline()
rapid_model.fit(x=gmm_ols.x, y=gmm_ols.y, chunks_no=chunks_no, warn_start=True)