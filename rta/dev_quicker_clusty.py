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
from itertools import islice

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

from time import time

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)

# AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
# 100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# more than one million free slots
# variables = ['rta', 'dta', 'massa']
# runs = np.array(list(set(A.run.unique()) | set(U.run.unique())))

# Aid = A.groupby('id')
# A_agg = pd.concat([ Aid[variables].median(),
#                     Aid.charge.first(), 
#                     pd.DataFrame(Aid.size(), columns=['signal_cnt'])], axis = 1)

# def iter_over_no_runs(A, runs, variables):
#     """Iterate over data not appearing in a given run.

#     This will not return a peptide that appeared in all the runs."""
#     Aid = A.groupby('id')
#     A_agg = pd.concat([ Aid[variables].median(), Aid.charge.first()], axis = 1)
#     A_short = A.loc[:,['id', 'run']]
#     for r in runs:
#         yield r, A_agg.loc[A_short[A_short.run != r].id.unique(),:]


def iter_over_rq(A, runs, variables):
    Aid = A.groupby('id')
    A_agg = pd.concat([ Aid[variables].median(), Aid.charge.first()], axis = 1)
    A_short = A.loc[:,['id', 'run']]
    for r in runs:
        d = A_agg.loc[A_short[A_short.run != r].id.unique(),:]
        for q, e in d.groupby('charge'):
            yield r, q, e[variables].sort_values('massa') # sorting!!! watch out!!!

# def iter_over_rq(A, runs, variables):
#     Aid = A.groupby('id')
#     A_agg = pd.concat([ Aid[variables].median(), Aid.charge.first()], axis = 1)
#     A_short = A.loc[:,['id', 'run']]
#     for r in runs:
#         d = A_agg.loc[A_short[A_short.run != r].id.unique(),:]
#         for q, e in d.groupby('charge'):
#             yield {'run':r, 'charge':q}, e[variables].sort_values('massa') # sorting!!! watch out!!!


# %%time
# s = iter_over_rq(A, runs, variables)
# S = list(s)

# r, q, Arq = next(iter_over_rq(A, runs, variables))
# add to index a mass range


## classification based on run an charge only:
# decided tp return the poitns from U that are closest to points in A.
# indices show, to whih points in A the points found in U correspond to.
def iter_run_charge_solution(A, runs, variables, max_radius=1):
    t0 = time()
    U_var = U.loc[:,variables]
    U_run_charge = U.loc[:,['run', 'charge']]
    print(time() - t0)
    out_cols = variables + ['u_id'] + [ "u_"+v for v in variables]
    for r, q, Arq in iter_over_rq(A, runs, variables):
        print(r, q)
        t0 = time()
        row_select = np.logical_and(U_run_charge.run == r, U_run_charge.charge == q)
        print(time() - t0)
        if np.any(row_select):
            t0 = time()
            Urq = U_var.loc[row_select,:]
            tree = kd_tree(Urq)
            # %%time
            # points = tree.query_ball_point(a, r=1, p=inf) # here the radius parametrizes the problem
            # %%time
            dist, points = tree.query(Arq, k=1, p=inf,
                                      distance_upper_bound=max_radius) # here: the maximal number of neighours
            out = Urq.iloc[points[dist < inf]].reset_index()
            out['d'] = dist[dist < inf]
            out['charge'] = q
            out['run'] = r
            out.index = Arq[dist < inf].index
            print(time() - t0)
            print()
            yield out


%%time
x = list(iter_run_charge_solution(A, runs, variables, max_radius=10))

# nearest_neighbours = pd.concat(x, axis=0, sort=False)
# plt.hist(nearest_neighbours.d, bins=200)
# plt.show()


# def iter_over_r(A, runs, variables):
#     Aid = A.groupby('id')
#     A_agg = pd.concat([ Aid[variables].median(), Aid.charge.first()], axis = 1)
#     A_short = A.loc[:,['id', 'run']]
#     for r in runs:
#         d = A_agg.loc[A_short[A_short.run != r].id.unique(),:]
#         yield r, d

def iter_over_rq_exp(A, runs, variables):
    Aid = A.groupby('id')
    A_agg = pd.concat([ Aid[variables].median(), Aid.charge.first()], axis = 1)
    A_short = A.loc[:,['id', 'run']]
    for r in runs:
        d = A_agg.loc[A_short[A_short.run != r].id.unique(),:]
        for q, e in d.groupby('charge'):
            to_query = "run == {} and charge == {}".format(r, q)
            to_eval  = "run = {}\ncharge = {}".format(r, q)
            Arq = e[variables].sort_values('massa') # sorting!!! watch out!!!
            yield to_query, Arq

to_query, A_cond = next(iter_over_rq_exp(A, runs, variables))


#TODO: change it so that freevars and condvars are explicity passed further on.
# optimize this code: for sure it's stupid.
# make a class of it.
def get_iter_over_rq(A, runs,
                     peptID='id',
                     condvars=['id', 'run'],
                     freevars=['rta', 'dta', 'massa']):
    def iter_over_rq_exp():
        Aid = A.groupby(peptID)
        A_agg = pd.concat([ Aid[freevars].median(), Aid.charge.first()], axis = 1)
        A_short = A.loc[:,condvars]
        for r in runs:
            d = A_agg.loc[A_short[A_short.run != r][peptID].unique(),:]
            for q, e in d.groupby('charge'):
                U_query = "run == {} and charge == {}".format(r, q)
                Arq = e[freevars]
                yield U_query, Arq
    return condvars, freevars, iter_over_rq_exp()


def find_nearest_neighbour_exp(A_iter, U, freevars, condvars, max_radius=1):
    t0 = time()
    # U_freevars = U.loc[:, freevars]
    # U_condvars = U.loc[:, condvars]
    # U_ = U.loc[:,condvars + freevars]
    U_ = U.loc[:,freevars + condvars]
    print(time() - t0)
    # U_ind = U.index
    # U_condvars = U_condvars.reset_index().set_index(condvars)
    # U_condvars
    # %%time
    # U_condvars[U_condvars.index.get_level_values('run') != 1]
    # %%time
    # U_freevars.loc[[U_ind[np.logical_and(U_condvars.run == 1, U_condvars.charge == 1)]]
    for cond, A_cond in A_iter:
        print(cond)
        t0 = time()
        # U_condvars.query(cond)
        # %%time
        # np.any()
        # np.logical_and()
        # %%time
        # U.query(cond)
        # %%timeit
        # U_cond = U_freevars.loc[U_condvars.query(cond).index]
        # %%timeit
        U_cond = U_.query(cond)
        print(time() - t0)
        if not U_cond.empty:
            t0 = time()
            tree = kd_tree(U_cond[freevars])
            print(time()-t0)
            t0 = time()
            dist, points = tree.query(A_cond,
                                      k=1,
                                      p=inf,
                                      distance_upper_bound=max_radius) # here: the maximal number of neighours
            print(time()-t0)
            out = U_cond.iloc[points[dist < inf]].reset_index()
            out['d'] = dist[dist < inf]
            out.index = A_cond[dist < inf].index
            print(time() - t0)
            print()
            yield out



class RunChargeGrouper(object):
    def __init__(self, A, U,
                 peptID = 'id',
                 freevars=['rta', 'dta', 'massa']):
        self.condvars = ['run', 'charge']
        self.freevars = freevars
        self.rq = A[self.condvars].drop_duplicates()
        self.Urq = U.loc[:, freevars + self.condvars].groupby(['run', 'charge'])
        self.Arq = A.loc[:, [peptID] + self.condvars].groupby(['run', 'charge'])
        Aid = A.groupby(peptID)
        A_agg = pd.concat([ Aid[freevars].median(), Aid.charge.first()], axis = 1)
        self.A_agg_q = A_agg.groupby('charge')

    def __iter__(self):
        """Iterate over medoids of identified peptides and corresponding unindetified signals."""
        for q, r_q in self.rq.groupby('charge'):
            # all identified peptides with charge q: medoids
            A_agg_q = self.A_agg_q.get_group(q)
            for r in r_q['run']:
                Arq = self.Arq.get_group((r,q))
                Urq = self.Urq.get_group((r,q))
                # get all peptide-medoids not in run r with charge q
                A_agg_rq = A_agg_q.loc[~A_agg_q.index.isin(Arq.id), self.freevars]
                if not A_agg_rq.empty:
                    yield A_agg_rq, Urq


def find_nearest_neighbour(grouper, max_radius=1):
    for A_cond, U_cond in grouper:
        tree = kd_tree(U_cond[grouper.freevars])
        dist, points = tree.query(A_cond, k=1, p=inf, distance_upper_bound=max_radius)
        out = U_cond.iloc[points[dist < inf]].reset_index()
        if not out.empty:
            out['d'] = dist[dist < inf]
            out.index = A_cond[dist < inf].index
            yield out

%%time
rq_grouper = RunChargeGrouper(A, U)
NN_rq = find_nearest_neighbour(rq_grouper, max_radius=1)
y = list(NN_rq)



# %%time
# NN_rq = iter_run_charge_solution(A, runs, variables, max_radius=1)
# x = list(islice(NN_rq, 5))


# 30 s on old computer for query_ball_point, 
# 20 s for a simple query about 10 neighbors. # this is not bad, provided that we would actually want one point

# Open quesiton: how much faster can this be done?
# need a rqm iterator for testing




## classification based on run an charge only:
mass_ppm_thr = 10
# mass_ppm_thr = 100
masses = d.massa.values

%%time
masses = np.sort(A.massa.values)
good_diffs = np.diff(masses)/masses[:-1]*1e6 > mass_ppm_thr
L = masses[np.insert(good_diffs, 0, True)]
R = masses[np.insert(good_diffs, -1, True)]
I = pd.IntervalIndex.from_arrays(L, R, closed = 'left')
x = pd.cut(masses, I)


masses



"""Idea here:
Query for the closest points in U next to the mediods calculated for peptides in A.
"""
# a forest of kd_trees this could be done multicore?
# maybe, but wait for the CV
# F = {run: kd_tree(U.loc[U.run == run, variables]) for run in runs}
# 5.37s on modern computer

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

# define few meaningful splits: based on mmu (mili-mass units)
# thr = '5mmu'
# parse_thr(thr)
# thr = '5da'
# parse_thr(thr)
# thr = '5ppm'
# parse_thr(thr)

# res = {}
# for r in runs:
#     res[r] = F[r].query(
#         A_agg.loc[A_agg.run.apply(lambda x: r not in x), variables],
#         p=inf,
#         k=1)

Did = D.groupby('id')
HB = pd.concat( [   get_hyperboxes(D, variables),
                    Did[variables].median(),
                    pd.DataFrame(Did.size(), columns=['signal_cnt']),
                    Did.run.agg(frozenset), # as usual, the slowest !!!
                    Did.charge.median(),
                    Did.FWHM.median()     ],
                    axis = 1)
HB = HB[HB.signal_cnt > 5]
# all values have been filtered so that only one charge state is used for the analysis
# Counter(A.groupby('id').charge.nunique())
# use this to divide the data

def in_closed_intervals(mz, left_mz, right_mz):
    i = np.searchsorted(right_mz, mz, side='left')
    out = np.full(mz.shape, -1)
    smaller = mz <= right_mz[-1]
    out[smaller] = np.where(np.take(left_mz, i[smaller]) <= mz[smaller],
                            i[smaller], -1)
    return out

in_closed_intervals(U.massa.values, L, R)

# Alternativ für Zukunft
# first sort the bloody U and then do calculations on views.
u_vars = ['run', 'charge', 'massa', 'rta', 'dta']

UU = U.loc[:,u_vars]

# UU = UU.sort_values(['run', 'charge', 'massa'])

%%time
UU_g = UU.groupby(['run', 'charge'])
UU_g.describe()

x = list(UU_g)

%%time
UU = U.loc[:,u_vars]
UU = UU.sort_values(['run', 'charge'])
UU = UU.set_index(['run', 'charge'])

UU.loc[(1,1),'massa']
UU.xs(1, level='run')

x = pd.DataFrame({'a': [10, 20], 'b':['a', 'b']},
                 index = pd.IntervalIndex.from_tuples([(0, 1), (3, 5)]))

x.loc[[4, 4.5, 5.5]]


# %%time
# F = kd_tree(U[variables])
## 36.7 seconds

# %%time
# F = {run: kd_tree(U.loc[U.run == run, variables]) for run in runs}
# ## 5.15 seconds

charges = np.array(list(set(U.charge.unique()) | set(A.charge.unique())))

%%time
F = {}
U_var = U.loc[:,variables]
for q in charges:
    for r in runs:
        row_select = np.logical_and(U.run == r, U.charge == q)
        F[(r,q)] = kd_tree(U_var.loc[row_select,:]) if np.any(row_select) else None
# 3.01 sec / 13 sec old





# U.sort_values(['run', 'charge', 'massa']) # this is lenghty.

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


