"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.align.calibrator       import Calibrator
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

    # add the preprocessing step for the calibrator for dt:
    # it should remove from the analysis proteins that are bloody repeating
    # in different charges.

from rta.align.calibrator import Calibrator


d = preprocess(annotated_all, min_runs_no,
               _get_stats = {'retain_all_stats': True})
d.all_stats.head()
d.D.head()
d.D.columns
# remove peptides with different charges across runs entirely
d.stats.head()
# d.filter_multiply_charged()

dpp = d.D[d.D[d.pept_id].isin(d.stats[d.stats.charges > 1].index)].copy()
dpp.columns
dpp.head(30)
dpp.to_csv('/home/matteo/Projects/retentiontimealignment/Data/multiply_charged_peptides.csv',
           index=False)



# remove peptides that do not follow correct pattern


# add desciption methods for preprocessing.
    # one idea: to have the information on how many peptides can be found in 1, 2, 3, and more runs.
    # this will come in handy: we want to know the data!



class DTcalibrator(Calibrator):
    def __init__(self,
                 preprocessed_data,
                 feature='dt',
                 folds_no=10):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pandas.DataFrame): data to assign folds to.
            feature (string): the name of the feature in the column space of the preprocessed_data that will be aligned.

        """
        self.folds_no = folds_no
        self.d = preprocessed_data
        self.d.filter_unfoldable_strata(self.folds_no)
        self.feature = feature
        self.feature_stat = feature + '_' + preprocessed_data.stat_name
        self.feature_stat_distance = self.feature_stat + '_distance'
        self.D = self.d.D
        self.pept_id = self.d.pept_id
        self.stats = self.d.stats
        self._trim_stats_and_D(self.stats.charges == 1)
        self.D = self.D[[self.d.pept_id,
                         self.d.run_name,
                         self.feature,
                         self.feature_stat_distance]]
        self.D.columns = ['id', 'run', 'x', 'y']

    def _trim_stats_and_D(self, retain):
        """Filter peptides that are not in 'self.stats.index'.

        Args:
            retain (np.array of logicals): retain these peptides within 'self.stats'
        """
        self.stats = self.stats[retain].copy()
        self.D = self.D[self.D[self.pept_id].isin(self.stats.index)].copy()

d = preprocess(annotated_all, min_runs_no,
               _get_stats = {'retain_all_stats': True})
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