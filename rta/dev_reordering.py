%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt

from rta.read.csvs import big_data
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run
from rta.reference import stat_reference

annotated_all, unlabelled_all = big_data()
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")
# D.fold will be used in CV, but this will be performed outside A.

# X, uX = choose_run(D, 'rt', 1)
# X, uX = choose_most_shared_run(D, 'rt', stats)
# X, uX = choose_statistical_run(D, 'rt', 'mean')

var2align = 'rt'

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

x1 = a(X)
X['x1'] = x1
# X = X.drop(['yhat'], 1)
# a.res()
# a.fitted()
# a.plot(s=1)
# a.plot(s=1, residuals=True)
# a.plot_residuals(s=1)
## Plot results of the coordinate models.
# a.m[1].plot(s=1)
# a.m[1].plot_residuals(s=1)

X = stat_reference(X[['run', 'x']], 'median')

def centiles(x):
    """Get centiles of x"""
    return np.quantile(x, [i/100 for i in range(101)])

# X = X.drop(['x1', 'y1'], 1)
# X.rename(columns={"x": "x0", "y": "y0"}, inplace=True)

# Maybe the model should be initialized?
def Tenzerize(X, n, a, stat='median'):
    """Perform a hunt for correct alignment."""
    for i in range(n):
        a.fit(X)
        x = a(X)
        X.rename(columns={'x':'x'+str(i), 'y':'y'+str(i)}, inplace=True)
        X['x'] = x
        X = stat_reference(X, stat)
    X.rename(columns={'x':'x'+str(n), 'y':'y'+str(n)}, inplace=True)
    return X

n = 4
X = Tenzerize(X, n, a)
for i in range(n+1):
    pass


def Matteotize(X, a, stat='median'):
    """Simply run one alignment once (maybe twice), but good."""
    pass





