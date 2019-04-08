%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from collections import Counter
from scipy.spatial import cKDTree as kd_tree
from plotnine import *

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)

AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# more than one million free slots

# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']

## simplify to peptide-medoids
Aid = A.groupby('id')
A_agg = pd.concat(	[Aid[variables].median(),
					 pd.DataFrame(Aid.size(), columns=['signal_cnt']),
					 Aid.run.agg(frozenset)],
					 axis = 1)
# counts = Counter(A_agg.signal_cnt)

A_agg_no_fulls = A_agg.loc[A_agg.signal_cnt != 10]
A_agg_no_fulls = A_agg_no_fulls.reset_index()
runs = np.unique(U.run)
"""Idea here:
Query for the closest points in U next to the mediods calculated for peptides in A.
"""
# a forest of kd_trees
# this could be done multicore ;)
F = {run: kd_tree(U.loc[U.run == run, ['massa', 'rta', 'dta']]) for run in runs}
# 5.37s on modern computer



a_agg_no_fulls = A_agg_no_fulls.iloc[[0,1,2,3]]
x = a_agg_no_fulls.iloc[0]

def nn_iter(x):
	peptID, p_rta, p_dta, p_mass, p_run_cnt, p_runs = x
	for r in runs:
		if not r in p_runs:
			d, idx = F[r].query([p_mass, p_rta, p_dta], p=inf, k=1)
			nn_mass, nn_rta, nn_dta = F[r].data[idx]
			yield (peptID, r, nn_mass, nn_rta, nn_dta, idx, d)

def fill_iter(X):
	for x in X.values:
		yield from nn_iter(x)
# %%time
# FILLED = pd.DataFrame(fill_iter(A_agg_no_fulls))
# variables = ['id', 'run', 'massa', 'rta', 'dta', 'idx', 'd']
# FILLED.columns = variables
# FILLED['origin'] = 'U'
# A_ = A.loc[:,variables[:-2] + ['signal2medoid_d']]
# A_ = A_.reset_index()
# names = list(A_.columns)
# names[0] = 'idx'
# A_.columns = names
# A_['origin'] = 'A'
# ALL = pd.concat([A_, FILLED], axis=0, sort=False)
# ALL.to_msgpack(data/"nnn_zlib.msg", compress='zlib') # naive nearest neighbours
A_agg = A_agg.sort_values('massa')
sum(np.diff(A_agg.massa) > .1)

# define few meaningful splits: based on mmu (mili-mass units)
thr = '5mmu'
parse_thr(thr)


thr = '5da'
parse_thr(thr)

thr = '5ppm'
parse_thr(thr)

w = A.groupby('id').massa_d.median()
w = w[w > 0]
sum(np.diff(A_agg.massa) > max(w))

x = A_agg.massa.values
max_allowed = 0.001
w = np.arange(len(x)-1)[np.diff(x) > max_allowed]

res = {}
for r in runs:
	res[r] = F[r].query(
		A_agg.loc[A_agg.run.apply(lambda x: r not in x), ['massa', 'rta', 'dta']],
		p=inf,
		k=1)
# to do: check for balls instead
# what is this box-size? Maybe it's precisely the thing I look for?
# make test
# also, maybe it is still a good idea to get the closest points in all directions
# to somehow estimate the noise level? No, this is to vague


