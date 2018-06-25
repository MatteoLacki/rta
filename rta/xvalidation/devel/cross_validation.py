"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter as count
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)

from rta.models.base_model import coef, predict, fitted, coefficients, residuals
from rta.models.plot import plot
from rta.models.SQSpline import SQSpline
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param

annotated_all, unlabelled_all = big_data()

folds_no = 10
min_runs_no = 5
annotated_cv, annotated_stats, runs_cnts = preprocess(annotated_all,
                                                      min_runs_no,
                                                      folds_no)
slim_features = ['id','run','rt','rt_median_distance','fold']
data = annotated_cv_slim = annotated_cv[slim_features]

cores_no = 16
data = annotated_cv_slim
Model = SQSpline
parameters = [{"chunks_no": n} for n in range(2,50)]



with Pool(cores_no) as p:
    results = p.starmap(cv_run_param, tasks_run_param(data, parameters))

