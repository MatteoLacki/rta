"""Gaussian Mixture Models and Ordinary Least squares.

An alternative to Huber regression.
Divide the X argument into percentiles.
On each group fit a GMM.
Obtain signal/noise and their variances.
Then fit OLS with a diagonal covariance matrix.
"""

import numpy as np
from patsy import dmatrices, ModelDesc
from sklearn.mixture import GaussianMixture as GM

def parse_formula(formula):
	"""Parse the formula.

	Retrieve names of response and control variables.
	"""
	response = formula.split('~')[0].replace(' ','')
	args = formula[formula.find('(')+1:formula.find(')')]
	args = args.replace(" ","").split(",")
	# check for main argument ("x=")
	control = ""
	for arg in filter(lambda x: "x=" in x, args):
		control = arg[arg.find("=")+1:]
	if not control:
		control = args[0]
	return control, response


def get_quantiles(x, chunks_no=4):
	"""A silly function."""
	return np.percentile(x, np.linspace(0, 100, chunks_no+1))


def get_k_percentile_pairs_of_N_integers(N, k):
	"""Generate pairs of consecutive k-percentiles of N first integers.

	The distribution concentrates on set { 0 , 1 , .., N-1 }.
	For k = 10, you will get indices of approximate deciles.
	"""
	assert N >= 0 and k > 0
	base_step = N // k
	res = N % k
	s = 0 # start
	e = 0 # end
	for _ in range(1, k+1):
		e += base_step
		if res > 0:
			e += 1
			res -= 1
		yield s, e
		s = e

# TODO: warm start?
# not so complicated to merge it if done from top to down
# the iteration of grid search should go from top to bottom
# 
# TODO: Grid search not so difficult in terms of GMMs.
class GMM_OLS(SplineRegression):
	def fit(self, formula, data={}, data_sorted=False, **kwds):
		self.formula = formula
		vars_names = control_name, response_name = parse_formula(formula)

		if not data_sorted: # we can avoid that up the call tree
			self.data = data = data.sort_values(by = list(vars_names))
		
		response = np.asarray(data[response_name]).reshape(-1,1)
		control = np.asarray(data[control_name]).reshape(-1,1)

		# here we might need np.asarray -> f****** pickle
		y, X = dmatrices(formula, data)
		self.y = y = y.ravel()
		self.X = X
		
	def __repr__(self):
		return "GMM_OLS"

chunks_no = 5

# this works on views
g_mix = GM(n_components = 2, warm_start=True)
g_mixes = [g_mix.fit(response[s:e]) for s, e in 
		   get_k_percentile_pairs_of_N_integers(len(control), chunks_no)]


