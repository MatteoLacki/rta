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

c.plot()

m = c.cal_res[0][2]
m.plot()
m.cv_stats

dt_cal = Calibrator(d, feature='dt', folds_no=folds_no)
dt_cal.fold()
dt_cal.calibrate()
dt_cal.plot()

dt_cal.cal_res[10][2].plot()

# add the preprocessing step for the calibrator for dt:
# it should remove from the analysis proteins that are bloody repeating
# in different charges.




