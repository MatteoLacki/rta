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
    # parameters = [{"chunks_no": n }for n in range(2,200)]
    # c.calibrate(parameters)
    c.plot()

    m = c.cal_res[0][2]
    m.plot()
    m.cv_stats
    # finish off the collection of stats for purpose of choosing
    # the best models


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

# TODO finish off the procedure is_signal for GaussianMixtureSpline

gms = GaussianMixtureSpline()
gms.fit(x, y, chunks_no = 20)
gms.signal
gms.plot()
gms.is_signal(np.array([10, 40]),
              np.array([10, 40]))


x_new = np.array([10, 40,  76,  -100, 200, 160])
y_new = np.array([1.0,4.0, 0.1, -100, 200, -.5])

gms.is_signal(x_new, y_new)

i = np.searchsorted(gms.x_percentiles, x_new) - 1
in_range = np.logical_and(i > -1, i < gms.chunks_no - 1)
signal = np.full(shape = x_new.shape,
                 fill_value = False,
                 dtype = np.bool_)
i = i[in_range]
y_new = y_new[in_range]
# we have made the roots to solve this problem. Use that function!!!
# To select the signal from noise.
bottom_lines = gms.signal_regions[i,0]
top_lines = gms.signal_regions[i,1]
signal[in_range] = (bottom_lines <= y_new) & (y_new <= top_lines)

# we can add this to the plot, to show how nice it works.

# dist_to_means = np.abs(gms.means[i, 0] - y_new[in_range])
# signal[in_range] = dist_to_means <= 
#     gms.sds[i] * gms.std_cnt


x_new[~indices_out_of_range]

# It still have to sort out the problem of these values...

# srs = []
# for i in range(gms.chunks_no):
#     m_s, m_n = gms.means[i,]
#     p_s, p_n = gms.probs[i,]
#     W, V = gms.variances[i,]
#     sd_s, sd_n = sqrt(W), sqrt(V)
#     sr = signal_region(m_s,
#                        m_n,
#                        sd_s,
#                        sd_n,
#                        p_s,
#                        p_n,
#                        simple_calc=True)
#     srs.append(sr)

rs = RobustSpline()
rs.fit(x,y,chunks_no=20)
rs.x_percentiles
c = rs.is_signal(x_new, y_new)
rs.plot(show=False)
plt.scatter(x_new, y_new, c=c)
plt.show()


