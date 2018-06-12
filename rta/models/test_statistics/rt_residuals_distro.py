"""Check the distribution of the error terms from one run of Huber regression.

This way we will ascertain, if only one step of Huber fitting is necessary.
Alternatively, we could refit on the 'signal' data with either L2 or L1 regressors.
Possibly checking for the values of some parameters.
"""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices
from sklearn.linear_model import HuberRegressor
from statsmodels import robust

from rta.models.base_model import cv, coef, predict
from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.xvalidation import grouped_K_folds, filter_foldable


# get data
annotated, unlabelled = big_data()
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
K = 5 # number of folds
annotated_cv = filter_foldable(annotated, annotated_stats, K)
folds = annotated_cv.groupby('runs').rt.transform(grouped_K_folds,
												  K = K).astype(np.int8)
annotated_cv = annotated_cv.assign(fold=folds)
annotated_cv_1 = annotated_cv[annotated_cv.run == 1]
data = annotated_cv_1

# fit the model
formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
y, X = map(np.asarray, dmatrices(formula, data))
y = y.ravel()


def fit_huber(epsilon=1.35, alpha=.001):
	h_reg = HuberRegressor(warm_start = True,
	                       fit_intercept = True,
	                       alpha=alpha,
	                       epsilon=epsilon)
	h_reg.fit(X, y)
	coefs = h_reg.coef_
	outliers = h_reg.outliers_
	signal = np.logical_not(outliers)
	fitted_values = np.dot(X, coefs)
	residuals = y - fitted_values
	signal_residuals = residuals[signal]
	return residuals, signal, outliers, h_reg


# make a plot of the distribution of the signal residuals
def make_residual_plots(plots_no = 10):
	fig, axs = plt.subplots(1, plots_no, sharey=True, tight_layout=True)
	for i, epsilon in enumerate(np.linspace(1, 3, plots_no)):
		r, s, o, m = fit_huber(epsilon=epsilon)
		signal_residuals = r[s]
		axs[i].hist(signal_residuals, bins = 100)
		axs[i].set_title(epsilon)
	plt.show()

plots_no = 10
make_residual_plots(plots_no)

hubers = [fit_huber(epsilon=epsilon) 
		  for i, epsilon in enumerate(np.linspace(1.1, 3, plots_no))]
	# there seems to be some bias towards smaller values
[(np.mean(r[s]), np.median(r[s])) for r,s,_,_ in hubers]



## refitting with least squares to the signal
def fit_ls(s):
	coefs, sum_of_res, rank, sing_vals = np.linalg.lstsq(a=X[s], b=y[s])
	coefs = coefs.ravel()
	fitted_values = np.dot(X[s], coefs)
	residuals = y[s] - fitted_values
	return residuals, np.mean(residuals), np.median(residuals)


def make_residual_plots_for_signals(plots_no):
	fig, axs = plt.subplots(1, plots_no, sharey=True, tight_layout=True)
	for i, epsilon in enumerate(np.linspace(1.1, 3, plots_no)):
		_, s, _, _ = fit_huber(epsilon=epsilon)
		residuals, mean, median = fit_ls(s)
		axs[i].hist(residuals, bins = 100)
		axs[i].set_title(epsilon)
	plt.show()

make_residual_plots_for_signals(plots_no)

