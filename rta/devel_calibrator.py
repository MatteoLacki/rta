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
c.d.stats

c.D


it = c.iter_run_param()
next(it)


def cv_run_param(run_no,
                 run_data,
                 param,
                 folds,
                 model=robust_spline):
    
D = c.d.D
D.columns
D[['id', 'run', ]]

def iter_run_param():



%%timeit
c.fold()