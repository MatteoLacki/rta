"""Develop the calibrator."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.read_in_data import big_data
from rta.pre.processing import preprocess
from rta.align.calibrator import Calibrator

folds_no = 10
min_runs_no = 5

annotated_all, unlabelled_all = big_data()
d = preprocess(annotated_all, min_runs_no)


c = Calibrator(d, feature='rt', folds_no=folds_no)
c.fold()


