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
%%timeit
c.calibrate()
# less that 1.3 seconds on default params. 


parameters = [{"chunks_no": n} for n in range(2,50)]

%%timeit
c.calibrate(parameters)



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