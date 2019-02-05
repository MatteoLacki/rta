%load_ext line_profiler
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture, cluster
from sklearn.model_selection import GroupKFold
from collections import Counter
from multiprocessing import Pool, Process
from patsy import dmatrices, dmatrix, bs

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot, plot_curve
# from rta.splines.denoising import denoise_and_align
# from rta.splines.denoising2 import denoise_and_align, denoise_and_align2, denoise_and_align_run, denoise_and_align_run2
from rta.misc import max_space


annotated, unlabelled = big_data(path = "rta/data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
annotated_slim = annotated[['run', 'rt', 'rt_median_distance']]
unlabelled_slim = unlabelled[['run', 'rt']]





model = 'Huber'
refit = True

run1 = annotated_slim[annotated_slim.run == 1]

import json
with open("rta/data/run1.json",'rb') as h:
	json.dump(run1, h)


bs_res = bs(run1.rt, 
		    df=40, 
		    degree=2, 
		    lower_bound=0, 
		    upper_bound=200, 
		    include_intercept=True)





def denoise_align_run(run_cnt):
    """Remove noise and align the retention times in a run."""
    a = annotated_slim.loc[annotated_slim.run == run_cnt,]
    u = unlabelled_slim.loc[unlabelled_slim.run == run_cnt,]
	
    # fit the spline
    model = spline(a, formula)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides.
    if refit:
        model = spline(a[signal], formula)

    a_rt_aligned = np.array(a.rt) - predict(model, rt=a.rt) 
    u_rt_aligned = np.array(u.rt) - predict(model, rt=u.rt) 

    return signal, a_rt_aligned, u_rt_aligned


def denoise_align(workers_cnt=10):
    """Denoise and align all runs."""
    with Pool(workers_cnt) as workers:
        res = workers.map(denoise_align_run, list(range(1, 11)))
    return res


if __name__ == "__main__":
	res = denoise_align()
	print(res)



