"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
# %load_ext line_profiler

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

# from rta.align.calibrator       import Calibrator, DTcalibrator
from rta.align.calibrator       import calibrate
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.models.splines.gaussian_mixture import gaussian_mixture_spline


if __name__ == "__main__":
    folds_no    = 10
    min_runs_no = 5
    annotated_all, unlabelled_all = big_data()
    d = preprocess(annotated_all, min_runs_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrate(feature     ='rt',
                  data        = d,
                  folds_no    = folds_no,
                  min_runs_no = min_runs_no, 
                  model       = gaussian_mixture_spline,
                  align_cnt   = 3)

    W = c.D[c.D.run == 1]
    W.head()
    plt.scatter(W.rt_1, W.runs_stat_dist_1)
    plt.show()
    c.best_models[1][1].plot()
    c.D.head()

c.plot_run_alignments(2,  max_alignments=3, point_size=.1)
c.plot_run_alignments(9,  max_alignments=3, point_size=.1)
c.plot_run_alignments(10, max_alignments=3, point_size=.1)


# investigating the distribution of distances to the 'median'.
# It still seems, that we should not recalculate the medians all 
# the time....
plt.hist(c.D.runs_stat_dist_0, bins=5000, color='red')
plt.hist(c.D.runs_stat_dist_1, bins=5000, color='blue')
# plt.hist(c.D.runs_stat_dist_2, bins=1000, color='green')
plt.show()

