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

from rta.align.calibrator       import calibrator
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.data.column_names      import vars_annotated, vars_unlabelled


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

for abs_path in absolute_paths:
    annotated   = pd.read_csv(join(abs_path, 'annotated_data.csv'),
                              usecols = vars_annotated)
    unannotated = pd.read_csv(join(abs_path, 'unannotated_data.csv'),
                              usecols = vars_unlabelled)
    d = preprocess(annotated, min_runs_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrator(folds_no, min_runs_no, d, 'rt', 1)
    calibrated_results.append(c)


c = calibrated_results[0]
c.best_models[1].plot()


#@@@@@ The Mystery of strange distances.
from collections import Counter
d.D.head()

def signs_cnt_diff(x):
    return sum(x > 0) - sum(x < 0)

def count_zeros(x):
    return sum(x == 0)

def count_pos(x):
    return sum(x > 0)

def count_neg(x):
    return sum(x < 0)

d.D.groupby('id').mass_median_distance.agg(())

signs_diffs = d.D.groupby('id').mass_median_distance.agg(signs_cnt_diff)
Counter(signs_diffs)
signs_diffs[signs_diffs == -2]
signs_zero = d.D.groupby('id').mass_median_distance.agg(count_zeros)

Counter(np.abs(np.abs(signs_diffs.values) - signs_zero.values))


# Plotting:
ax1 = plt.subplot(2, 1, 1)
calibrated_results[0].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
calibrated_results[1].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.show()



ax1 = plt.subplot(2, 1, 1)
calibrated_results[2].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
calibrated_results[3].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.show()



ax1 = plt.subplot(2, 1, 1)
calibrated_results[0].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
calibrated_results[3].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.show()



# all runs from one given experiment.
ax1 = plt.subplot(6, 1, 1)
calibrated_results[0].best_models[1].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(6, 1, 2, sharex=ax1, sharey=ax1)
calibrated_results[0].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(6, 1, 3, sharex=ax1, sharey=ax1)
calibrated_results[0].best_models[3].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(6, 1, 4, sharex=ax1, sharey=ax1)
calibrated_results[0].best_models[1].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(6, 1, 5, sharex=ax1, sharey=ax1)
calibrated_results[0].best_models[2].plot(plt_style = 'seaborn-white', show=False)
plt.subplot(6, 1, 6, sharex=ax1, sharey=ax1)
calibrated_results[0].best_models[3].plot(plt_style = 'seaborn-white', show=False)
plt.show()


# Ute wants the old type of the plot: with medians on the x-axis