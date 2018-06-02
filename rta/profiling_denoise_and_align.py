%load_ext line_profiler

import numpy as np
import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.splines.denoising2 import denoise_and_align, denoise_and_align_run

annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
model = 'Huber'
refit = True

def iter_groups():
    for run_no, a in annotated.groupby('run'):
        u = unlabelled[unlabelled.run == run_no]
        yield a, u, formula, model, refit

a, u, formula, model, refit = next(iter_groups())

%lprun -f denoise_and_align_run denoise_and_align_run(a, u, formula, model, refit)
