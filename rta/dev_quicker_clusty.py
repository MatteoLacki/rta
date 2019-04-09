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

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)

AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
# 100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# more than one million free slots

# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']

## simplify to peptide-medoids
Aid = A.groupby('id')
A_agg = pd.concat(  [Aid[variables].median(),
                     pd.DataFrame(Aid.size(), columns=['signal_cnt']),
                     Aid.run.agg(frozenset)],
                     axis = 1)
# counts = Counter(A_agg.signal_cnt)    # number of peptides that occur a given number of times
# [(k, k*v) for k, v in counts.items()] # number of signals

A_agg_no_fulls = A_agg.loc[A_agg.signal_cnt != 10]
A_agg_no_fulls = A_agg_no_fulls.reset_index()
runs = np.unique(U.run)


"""Idea here:
Query for the closest points in U next to the mediods calculated for peptides in A.
"""
# a forest of kd_trees this could be done multicore?
# maybe, but wait for the CV
# F = {run: kd_tree(U.loc[U.run == run, variables]) for run in runs}
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
# A_agg = A_agg.sort_values('massa')
# sum(np.diff(A_agg.massa) > .1)

# define few meaningful splits: based on mmu (mili-mass units)
# thr = '5mmu'
# parse_thr(thr)
# thr = '5da'
# parse_thr(thr)
# thr = '5ppm'
# parse_thr(thr)

# w = A.groupby('id').massa_d.median()
# w = w[w > 0]
# sum(np.diff(A_agg.massa) > max(w))

# x = A_agg.massa.values
# max_allowed = 0.001
# w = np.arange(len(x)-1)[np.diff(x) > max_allowed]

# res = {}
# for r in runs:
#     res[r] = F[r].query(
#         A_agg.loc[A_agg.run.apply(lambda x: r not in x), variables],
#         p=inf,
#         k=1)
# to do: check for balls instead
# what is this box-size? Maybe it's precisely the thing I look for?
# make test
# also, maybe it is still a good idea to get the closest points in all directions
# to somehow estimate the noise level? No, this is to vague

from rta.array_operations.dataframe_ops import get_hyperboxes

Did = D.groupby('id')
HB = pd.concat( [   get_hyperboxes(D, variables),
                    Did[variables].median(),
                    pd.DataFrame(Did.size(), columns=['signal_cnt']),
                    Did.run.agg(frozenset),
                    Did.charge.median(),
                    Did.FWHM.median()     ],
                    axis = 1)
HB = HB[HB.signal_cnt > 5]

plt.hexbin(HB.rta_min, HB.rta_edge)
plt.hexbin(np.log(HB.rta_min), np.log(HB.rta_edge))
plt.show()
plt.hexbin(HB.dta_min, HB.dta_edge)
plt.hexbin(np.log(HB.dta_min), np.log(HB.dta_edge))
plt.show()
(ggplot(HB, aes(x='dta_min', y='dta_edge')) +
    geom_density_2d() +
    facet_wrap('charge'))
# (ggplot(HB, aes(x='FWHM', y='dta_edge')) +
#   geom_density_2d() +
#   facet_wrap('charge'))
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d())
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d() + facet_wrap('signal_cnt'))
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d() +
    facet_wrap('charge'))
(ggplot(HB, aes(x='massa_edge', group='charge', color='charge')) +
    geom_density())

# all values have been filtered so that only one charge state is used for the analysis
Counter(A.groupby('id').charge.nunique())

# %%time
# F = kd_tree(U[variables])
## 36.7 seconds

# %%time
# F = {run: kd_tree(U.loc[U.run == run, variables]) for run in runs}
# ## 5.15 seconds

charges = np.array(list(set(U.charge.unique()) | set(A.charge.unique())))

# %%time
# F = {}
# U_var = U.loc[:,variables]
# for q in charges:
#     for r in runs:
#         row_select = np.logical_and(U.run == r, U.charge == q)
#         F[(r,q)] = U_var.loc[row_select,:] if np.any(row_select) else None
# # 3.01 sec

%%time
F = {}
U_var = U.loc[:,variables]
for x in U[['run', 'charge']].drop_duplicates().itertuples():
    r, q = x.run, x.charge
    row_select = np.logical_and(U.run == r, U.charge == q)
    F[(r,q)] = kd_tree(U_var.loc[row_select,:])
# 2.44 sec: twice faster
# OK, the more subselection, the faster the construction of the kd-tree

#### GREAT!!!! So lets divide it all by the masses!!!! GREAT!!!
M = U.loc[np.logical_and(U.run == 1, U.charge == 2), 'massa'].values

M = np.sort(M)
dM = np.diff(M)
sum(dM > .1)


L = M - .1
R = M + .1




plt.hist(np.log(dM[dM > 0]), bins=100)
plt.show()

w = w[w > 0]
sum(np.diff(A_agg.massa) > max(w))
U.sort_values(['run', 'mass'])










# HB_long = pd.melt(HB[[v+'_edge' for v in variables]])
HB_long = pd.melt(HB[['signal_cnt', 'rta_edge', 'dta_edge', 'massa_edge']], id_vars='signal_cnt')

np.percentile(HB.rta_edge/np.percentile(HB.rta_edge, .99), .95)

lim_rect = HB[['rta_edge', 'dta_edge', 'massa_edge']].apply(lambda x: np.percentile(x, .99))
# lim_rect = np.log(HB[['rta_edge', 'dta_edge', 'massa_edge']]).apply(lambda x: np.percentile(x, .99))
W = HB[['rta_edge', 'dta_edge', 'massa_edge']]/lim_rect
X = pd.melt(W)

# these look more or less independent: but maybe an analysis of some other metric would be more sensible?
(ggplot(HB, aes(x='dta_edge', y='rta_edge')) + 
    geom_density_2d() + 
    coord_fixed() + 
    scale_y_log10() +
    scale_x_log10())

(ggplot(HB, aes(x='massa_edge', y='rta_edge')) + 
    geom_density_2d() + 
    coord_fixed() + 
    scale_y_log10() +
    scale_x_log10())

(ggplot(HB, aes(x='massa_edge', y='dta_edge')) + 
    geom_density_2d() + 
    coord_fixed() + 
    scale_y_log10() +
    scale_x_log10())

(ggplot(X, aes(x='value', color='variable', group='variable')) + geom_density())



# applying the normalization for test.
lim_rect.index = pd.Index(variables)
variables_n = [v+'_n' for v in variables]
U[variables_n] = U[variables]/lim_rect




HB_long.groupby('variable').value.apply(lambda x: (np.median(x), np.percentile(x, .99)))
HB_long.groupby('variable').value



for var in ['rta', 'dta', 'massa']:
    nor = get_normalization(A, var)
    for X in [A, U, D]:
        normalize(X, var, nor)

from plotnine import *

(ggplot(HB, aes(x='np.log(vol)')) + geom_density() + facet_wrap('signal_cnt'))



HB_long


