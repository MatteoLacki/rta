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


data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!

# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']


# almost 4 MLN points...
# AU = pd.concat([ A[variables], U[variables] ], axis=0)
# import hdbscan
# clusterer = hdbscan.HDBSCAN()
# clusterer.fit(AU) # takes FOREVER!!!!
# how to make it quicker?
# somehow arrange the points


## simplify to peptide-medoids
Aid = A.groupby('id')
A_agg = pd.concat([
	Aid[variables].median(),
	Aid.size(),
	Aid.run.agg(frozenset)
], axis=1)
var_names = [v + "_med" for v in variables]
var_names.extend(['signal_cnt', 'runs'])
A_agg.columns = var_names
# Counter(A_agg.signal_cnt)
## There are vastly more peptides appearing at only one run out of ten then any other
# This points to what? That the neighbourhood of such points should be a really nice cluster
# we should be ultra-conservative
A_agg_no_fulls = A_agg.loc[A_agg.signal_cnt != 10]
A_agg_no_fulls = A_agg_no_fulls.reset_index()


# I need more data-sets to check, if it all works as shown.
# I need to parse the xmls myselfs to bypass IsoQuant imports and the dependence upon MariaDB.
# Or, I might use it to my advantage: set up a database to store the parsed projects.
runs = np.unique(U.run)


"""Idea here:

Query for the closest points in U next to the mediods calculated for peptides in A.
Every 
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
			yield (peptID, r, nn_mass, nn_rta, nn_dta, idx)

def fill_iter(X):
	for x in X.values:
		yield from nn_iter(x)

FILLED = pd.DataFrame(fill_iter(A_agg_no_fulls))
variables = ['id', 'run', 'massa', 'rta', 'dta', 'u_idx']
FILLED.columns = variables



A[variables[:-1]]


# design a multiprocessor version of it???
#



