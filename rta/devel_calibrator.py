"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

# from rta.align.calibrator       import Calibrator, DTcalibrator
from rta.align.calibrator       import calibrator
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess, filter_unfoldable

if __name__ == "__main__":
    folds_no    = 10
    min_runs_no = 5
    annotated_all, unlabelled_all = big_data()
    d = preprocess(annotated_all, min_runs_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrator(folds_no, min_runs_no, d, 'rt', 3)

    W = c.D[c.D.run == 1]
    W.head()
    plt.scatter(W.rt_1, W.runs_stat_dist_1)
    plt.show()
    c.best_models[1].plot()
