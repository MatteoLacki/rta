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


data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)

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
Every fucking thing.
"""
# a forest of kd_trees
F = {run: kd_tree(U.loc[U.run == run, ['mass', 'rta', 'dta']]) for run in runs}
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

%%time
FILLED = pd.DataFrame(fill_iter(A_agg_no_fulls))
variables = ['id', 'run', 'massa', 'rta', 'dta', 'idx', 'd']
FILLED.columns = variables
FILLED['origin'] = 'U'

A_ = A.loc[:,variables[:-2] + ['signal2medoid_d']]
A_ = A_.reset_index()
names = list(A_.columns)
names[0] = 'idx'
A_.columns = names
A_['origin'] = 'A'

ALL = pd.concat([A_, FILLED], axis=0, sort=False)
ALL.to_msgpack(data/"nnn_zlib.msg", compress='zlib') # naive nearest neighbours

##### Accelaration






#####
from itertools import islice
ALL_id = islice(ALL.groupby('id').__iter__(), 30)
d = pd.concat((b for a, b in ALL_id), axis=0)

(ggplot(d, aes(x='rta', y='dta', color='origin', label='run')) +
	geom_text() +
	facet_wrap('id') +
	theme_dark())


# design a multiprocessor version of it???

# divide the data into subproblems and see, how much faster it can work?

# check, on the existing solution, how it looks like.
	# remember, the one and only real way to test it, is to x-validate the whole process
	# but for this, we need speed.


# plotting the log-volume of the boxes containing the points
def get_volumes(X, vars):
	for v in vars:
		X[v+'_v'] = X[v+'_max'] - X[v+'_min']


vars2 = ['massa', 'rta', 'dta']
HB = get_hyperboxes(ALL, vars2, 'id')
get_volumes(HB, vars2)
A_agg['logV'] = np.log(HB[[c for c in HB.columns if "_v" in c]]).sum(axis=1)
plt.hist(A_agg.logV, bins=300)
plt.show()

plt.style.use('default')

(ggplot(A_agg, aes(x='logV')) + 
	geom_histogram() +
	theme_lines())

(ggplot(A_agg, aes(x='logV')) + 
	geom_histogram() +
	facet_wrap('signal_cnt', scales='free_y'))

HBA = get_hyperboxes(A, vars2, 'id')
get_volumes(HBA, vars2)

A_agg['logV_0'] = np.log(HBA[[c for c in HB.columns if "_v" in c]]).sum(axis=1)
A_agg_non_inf = A_agg.loc[A_agg.logV_0 != -inf,]

(ggplot(A_agg_non_inf, aes(x='logV_0')) + 
	geom_histogram() +
	facet_wrap('signal_cnt', scales='free_y'))

(ggplot(A_agg_non_inf, aes(x='rta_med', y='dta_med', color='logV_0', size='logV_0')) +
	geom_point() +
	scale_size(range = (.1, 1)))

# How many points are shared?
idx_cnt = ALL.loc[ALL.origin != 'A',].groupby('idx').size()
peptide_sharing = Counter(idx_cnt)
non_shared_peptides_perc = peptide_sharing[1]/sum(v for k,v in peptide_sharing.items() if k > 1)

# find the farthest distance to the medoid in each group
Aidit = Aid.__iter__()
g, d = next(next(next(Aidit)))
d = d[vars2]

A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)
Aid = A.groupby('id')
A_agg['radius'] = Aid.signal2medoid_d.max()
A_agg_pos_radius = A_agg.loc[A_agg.radius > 0]

(	ggplot(A_agg_pos_radius, aes(x='radius')) + 
	geom_histogram(bins=100) + 
	facet_wrap('signal_cnt')	)

(	ggplot(A_agg_pos_radius, aes(x='radius**2')) + 
	geom_histogram(bins=100) + 
	facet_wrap('signal_cnt', scales='free_y')	)

(	ggplot(ALL.loc[ALL.d > 0,], aes(x='np.log(d)')) + 
	geom_histogram(bins=100) + 
	facet_wrap('origin', scales='free_y')	)

