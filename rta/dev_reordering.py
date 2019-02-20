%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt

from rta.plot.runs import plot_distances_to_reference
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run
from rta.reference import stat_reference

unlabelled_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')

D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")
# D.fold will be used in CV, but this will be performed outside A.

# X, uX = choose_run(D, 'rt', 1)
# X, uX = choose_most_shared_run(D, 'rt', stats)
# X, uX = choose_statistical_run(D, 'rt', 'mean')
var2align = 'rt'
X, uX = choose_statistical_run(D, 'rt', 'median')
plot_distances_to_reference(X, 'ggplot', s=1)

from rta.models.model import Model
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedian
from rta.math.stats import centiles

runs = D.run.unique()
m = {r: RollingMedian() for r in runs} # each run can have its own model
#TODO: add backfitting models.

a = Aligner(m)
a.fit(X)
# a.plot(plt_style='default') # works!
# a.plot(plt_style='ggplot', s=1) # works! All themes work.

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
# X = stat_reference(X[['run', 'x']], 'median')
# X = X.drop(['x1', 'y1'], 1)
# X.rename(columns={"x": "x0", "y": "y0"}, inplace=True)

from rta.align.strategies import Tenzerize

n = 4
X_tenzer = Tenzerize(X, n, a)
for i in range(n+1):
    pass

def Matteotize(X, a, stat='median'):
    """Simply run one alignment once (maybe twice), but good."""
    pass

# median extrapolation fit
from rta.models.rolling_median import RollingMedianSpline

X, uX = choose_statistical_run(D, 'rt', 'median')
rmi = RollingMedianSpline()
m = {r: RollingMedianSpline() for r in runs} # each run can have its own model
a = Aligner(m)
a.fit(X)
a.plot(s=1)
# This doesn't look good: maybe it should be applied to second order of error?
# from rta.models.denoiser import DenoiserRollingOrder
# x = X.x[X.run == 1].values
# y = X.y[X.run == 1].values
# d = y - x
# dro = DenoiserRollingOrder(l=6, w=np.ones(101), u=95, n=100)
# dro.fit(x, d)
# dro.plot()

# how to do backfitting properly?
# it does sound like an operation on the model.

from rta.models.backfit import Backfit
brmi = Backfit(rmi, 2)
brmi.fit(x, y-x)
brmi.plot_all()
brmi.plot()

X, uX = choose_statistical_run(D, 'rt', 'median')
m = {r: RollingMedianSpline() for r in runs} # each run can have its own model

