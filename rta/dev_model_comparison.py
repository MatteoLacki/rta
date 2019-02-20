%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt

from rta.plot.runs import plot_distances_to_reference
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run
from rta.reference import stat_reference
from rta.models.model import Model
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedian
from rta.math.stats import centiles


unlabelled_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')

D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)

## Getting reference run (y) for 'rt' alignment.
# X, uX = choose_run(D, 'rt', 1)
# X, uX = choose_most_shared_run(D, 'rt', stats)
# X, uX = choose_statistical_run(D, 'rt', 'mean')
X, uX = choose_statistical_run(D, 'rt', 'median')
# plot_distances_to_reference(X, 'ggplot', s=1)

runs = D.run.unique()
m = {r: RollingMedian() for r in runs} # each run can have its own model
a = Aligner(m)
a.fit(X)
a.plot(s=1)
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



from rta.models.backfit import Backfit
brmi = Backfit(rmi, 2)
brmi.fit(x, y-x)
brmi.plot_all()
brmi.plot()
