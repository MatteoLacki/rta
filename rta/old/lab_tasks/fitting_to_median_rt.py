"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import namedtuple
from os.path     import join as join

from rta.align.calibrator       import calibrate
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.data.column_names      import vars_annotated
from rta.models.splines.robust import robust_spline

Info = namedtuple('Info', 'flowtype time year month day')
try:
    data_path  = '~/Projects/Nano_VS_Micro/data/micro_vs_nano'
    file_names = os.listdir(data_path)
except FileNotFoundError:
    data_path  = '/Users/matteo/Projects/Nano_VS_Micro/data/micro_vs_nano'
    file_names = os.listdir(data_path)

files_desc = [Info('Microflow', 60,  2018, 7, 23),
              Info('Nanoflow',  60,  2018, 7, 17),
              Info('Microflow', 120, 2018, 7, 16),
              Info('Nanoflow',  120, 2018, 7, 16)]
#assume that each experiment consists of 6 runs and it is good enough
#for a protein to appear in 3 out of 6 runs.
# choosing min_runs_no out of 6.
min_runs_no        = 6
folds_no           = 10
absolute_paths     = [join(data_path, fn) for fn in file_names]
calibrated_results = []

# get medians, fit "rt - Me(rt) ~ Me(rt)", save to csv.
for abs_path in absolute_paths:
    annotated = pd.read_csv(join(abs_path, 'annotated_data.csv'),
                            usecols = vars_annotated)
    d = preprocess(annotated, min_runs_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrate(feature     = 'rt',
                  data        = d,
                  folds_no    = folds_no,
                  min_runs_no = min_runs_no, 
                  align_cnt   = 0)
    res = np.zeros(shape = c.D.run.values.shape)
    for i in np.unique(c.D.run):
        run_indices = c.D.run == i
        x = c.D.runs_stat_0[run_indices].values
        y = c.D.runs_stat_dist_0[run_indices].values
        res[run_indices] = y - robust_spline(x, y, 50)(x)
    D = c.D[['id', 'run', 'rt_0', 'runs_stat_0']].copy()
    D.columns = ['id', 'run', 'rt', 'rt_median']
    D['residual'] = res
    D.to_csv(path_or_buf = join(abs_path, "residuals.csv"),
             index       = False)

