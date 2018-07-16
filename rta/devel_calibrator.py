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


if __name__ == "__main__":
    folds_no    = 10
    min_runs_no = 5
    annotated_all, unlabelled_all = big_data()
    d = preprocess(annotated_all, min_runs_no,
                   _get_stats = {'retain_all_stats': True})
    c = Calibrator(d, feature='rt', folds_no=folds_no)
    c.fold()
    c.calibrate()
    # less that 1.3 seconds on default params. 
    # c.results[0].plot()
    # parameters = [{"chunks_no": n} for n in range(2,200)]
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
    # 

    dt_c = DTcalibrator(d, feature='dt', folds_no=folds_no)
    dt_c.fold()
    dt_c.calibrate()
    # less that 1.3 seconds on default params. 
    # c.results[0].plot()
    # parameters = [{"chunks_no": n} for n in range(2,200)]
    # c.calibrate(parameters)
    dt_c.plot()


    m = dt_c.cal_res[0][2]
    m.plot()


c.cal_res

from rta.models.splines.gaussian_mixture import GaussianMixtureSpline
from rta.models.splines.robust import RobustSpline

R1 = c.D[c.D.run == 1].copy()
x = R1.x.values
# x.shape = (x.shape[0], 1)
y = R1.y.values
# y.shape = (y.shape[0], 1)



dedup_sort(x, y)


gms = GaussianMixtureSpline()
gms.fit(x, y, chunks_no = 20)
gms.signal
gms.is_signal(np.array([10, 40]),
              np.array([10, 40]))

gms.x_percentiles


gms = GaussianMixtureSpline()
gms.fit(x, y, chunks_no = 20)
gms.signal
gms.is_signal(np.array([10, 40]),
              np.array([10, 40]))

gms.x_percentiles
gms.means[,0]




rs = RobustSpline()
rs.fit(x,y,chunks_no=20)
rs.x_percentiles

x.shape