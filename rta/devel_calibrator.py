"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.align.calibrator       import Calibrator, DTcalibrator
from rta.align.calibrator       import NeoCalibrator
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.models.splines.robust  import robust_spline


if __name__ == "__main__":
    folds_no    = 10
    min_runs_no = 5
    annotated_all, unlabelled_all = big_data()
    # d = preprocess(annotated_all, min_runs_no,
    #                _get_stats = {'retain_all_stats': True})
    d = preprocess(annotated_all, min_runs_no)
    d = filter_unfoldable(d, folds_no)    
    nc = NeoCalibrator(d, feature='rt', folds_no=folds_no)
    nc.runs_statistic()


    c = Calibrator(d, feature='rt', folds_no=folds_no)
    c.fold()
    c.calibrate()
    # less that 1.3 seconds on default params. 
    # c.results[0].plot()
    # parameters = [{"chunks_no": n }for n in range(2,200)]
    # c.calibrate(parameters)
    c.plot()
    c.select_best_models()
    m = c.best_models[1]
    m.plot()

    # finish off the collection of stats for purpose of choosing
    # the best models
    dt_cal = Calibrator(d, feature='dt', folds_no=folds_no)
    dt_cal.fold()
    dt_cal.calibrate()
    dt_cal.plot()
    dt_cal.cal_res[10][2].plot()
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


# should the calibrator have a routing for multiple fitting?
# it can use the same folding then: convenient and saves time.
# maybe we could even save on sorting values?

# problem: recalculate the statistics of the preprocessed data 



