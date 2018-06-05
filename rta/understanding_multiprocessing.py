%load_ext line_profiler
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture, cluster
from collections import Counter

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot, plot_curve
# from rta.splines.denoising import denoise_and_align
from rta.splines.denoising2 import denoise_and_align, denoise_and_align2, denoise_and_align_run, denoise_and_align_run2
from rta.misc import max_space


annotated, unlabelled = big_data(path = "rta/data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
annotated_slim = annotated[['run', 'rt', 'rt_median_distance']]
unlabelled_slim = unlabelled[['run', 'rt']]

model = 'Huber'
refit = True

def itergroups():
    for run_no, a in annotated.groupby('run'):
        u = unlabelled[unlabelled.run == run_no]
        yield a, u, formula, model, refit

a, u, formula, model, refit = next(itergroups())
signal, a_rt_aligned, u_rt_aligned = denoise_and_align_run(a, u, formula, model, refit)


%%timeit
res = denoise_and_align(annotated_slim,
                        unlabelled_slim,
                        formula,
                        workers_cnt=10)


## data to validate hypothesis about charge reductions and others.
# res.to_csv("~/Projects/retentiontimealignment/Data/rt_denoised.csv",
#            index = False)
model = "Huber"
refit = True

def iter_groups():
    for run_no, a in annotated.groupby('run'):
        u = unlabelled[unlabelled.run == run_no]
        yield a, u, formula, model, refit

a, u, formula, model, refit = next(iter_groups())



annotated_slim[annotated_slim.run == 1]

res = denoise_and_align2(annotated_slim,
                         unlabelled_slim,
                         runs_no=10,
                         formula=formula,
                         workers_cnt=10)


%%timeit
denoise_and_align_run2(1,
				       annotated_slim,
                       unlabelled_slim,
                       formula,
                       model)


%lprun -f denoise_and_align_run2 denoise_and_align_run2(1, annotated_slim, unlabelled_slim, formula, model)
%lprun -f denoise_and_align_run2 denoise_and_align_run2(1, annotated, unlabelled, formula, model)