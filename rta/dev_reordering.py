%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)

from rta.read.csvs import big_data
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run

annotated_all, unlabelled_all = big_data()
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")
# D.fold will be used in CV, but this will be performed outside A.

# X, uX = choose_run(D, 'rt', 1)
# X, uX = choose_most_shared_run(D, 'rt', stats)
# X, uX = choose_statistical_run(D, 'rt', 'mean')
X, uX = choose_statistical_run(D, 'rt', 'median')

from rta.models.model import Model
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedian

runs = D.run.unique()
m = {r: RollingMedian() for r in runs} # each run can have its own model
#TODO: add backfitting models.

a = Aligner(m)
a.fit(X)
# a.plot(plt_style='default') # works!
# a.plot(plt_style='ggplot') # works! All themes work.

y = a(X)
X['yhat'] = y
# X = X.drop(['yhat'], 1)
# a.res()
# a.fitted()
# a.plot(s=1)
# a.plot(s=1, residuals=True)
# a.plot_residuals(s=1)
## Plot results of the coordinate models.
# a.m[1].plot(s=1)
# a.m[1].plot_residuals(s=1)

plt.scatter()
a.m[1].x
a.m[1].res()

X.index.name

def get_distances_to_

if D.index.name:
	print('s')
else:
	print('b')

choose_statistical_run(X, )

import numpy as np

def stat_reference(X, stat='median', ref_name='y1'):
	assert stat in ('median', 'mean')
	ref = X.groupby('id').x.median()
	ref.name = ref_name
	

X.groupby('id').x.median().loc['YYVTI NA']
np.median(X.loc['YYVTI NA'].x.values)
