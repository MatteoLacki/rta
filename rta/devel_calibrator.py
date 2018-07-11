"""Develop the calibrator."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.align.calibrator import Calibrator
from rta.read_in_data import big_data
from rta.pre.processing import preprocess
from rta.models.splines.robust import robust_spline

folds_no    = 10
min_runs_no = 5

annotated_all, unlabelled_all = big_data()
d = preprocess(annotated_all, min_runs_no)

c = Calibrator(d, feature='rt', folds_no=folds_no)
c.fold()
c.calibrate()

# less that 1.3 seconds on default params. 
# c.results[0].plot()
parameters = [{"chunks_no": n} for n in range(2,200)]
# c.calibrate(parameters)

r, p, m = c.cal_res[10]
m.plot()
m.cv_stats
m.fold_stats


m.cv_stats.loc['std', 'fold_mad']


c.plot()

from collections import defaultdict

parameters = c.parameters
opt_var = 'chunks_no'
opt_var_vals = sorted([p[opt_var] for p in parameters])

mad_mean = defaultdict(list)
mad_std  = defaultdict(list)
for r, p, m in c.cal_res:
    cvs = m.cv_stats
    mad_mean[r].append(cvs.loc['mean', 'fold_mae'])
    mad_std[r].append(cvs.loc['std', 'fold_mae'])


for r in c.d.runs:
    x, y = opt_var_vals, mad_mean[r]
    plt.plot(x, y, label=r)
    plt.text(x[ 0], y[ 0], 'Run {}'.format(r))
    plt.text(x[-1], y[-1], 'Run {}'.format(r))

plt.show()



def cv_run_param(r, x, y, f, p):
    m = robust_spline(x, y,
                      drop_duplicates_and_sort=False,
                      folds = f,
                      **p)
    return m

with Pool(16) as p:
    res = p.starmap(cv_run_param, it)


# add simple visualization to Calibrator.
# 





%%timeit
c.fold()