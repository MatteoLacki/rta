%load_ext line_profiler
%load_ext autoreload
%autoreload 2

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
annotated, unlabelled = big_data(path = "rta/data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
annotated_slim = annotated[['run', 'rt', 'rt_median_distance']]
run1 = annotated_slim[annotated_slim.run == 1]


import numpy as np
from rta.models.base_model import Model
from rta.models.sklearn_regressors import SklearnRegression
from sklearn.model_selection import cross_val_score

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

test = SklearnRegression()
test.fit(formula, run1, warm_start=True)


from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
scoring = {'AUC': 'roc_auc'}


cross_val_score(test.regressor, 
				np.asarray(test.X), 
				np.asarray(test.y).flatten(), 
				cv=10,
				n_jobs=-1,
				scoring='neg_mean_squared_error')
