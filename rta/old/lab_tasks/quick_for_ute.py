"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.align.calibrator       import Calibrator, DTcalibrator
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess
from rta.models.splines.robust  import robust_spline



annotated = pd.read_csv('~/Projects/Nano_VS_Micro/data/annotated_data.csv',
                        usecols = vars_annotated)
unannotated = pd.read_csv('~/Projects/Nano_VS_Micro/data/unannotated_data.csv',
                          usecols = vars_unlabelled)

min_runs_no = 3
folds_no    = 10

d = preprocess(annotated, min_runs_no,
               _get_stats = {'retain_all_stats': True})
c = Calibrator(d, feature='rt', folds_no=folds_no)
c.fold()
c.calibrate()
# less that 1.3 seconds on default params. 
# c.results[0].plot()
# parameters = [{"chunks_no": n }for n in range(2,200)]
# c.calibrate(parameters)
c.plot()
c.select_best_models()


c.best_models[2].plot(plt_style = 'seaborn-white')

for i in range(6):
    plt.subplot(, 1, i)
    c.best_models[i-1].plot(show=False)

plt.show()