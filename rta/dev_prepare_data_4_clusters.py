%load_ext autoreload
%autoreload 2

from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rta.read.csvs import big_data
from rta.reference import cond_medians
from rta.preprocessing import preprocess
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedian
from rta.math.stats import med_mad, robust_chebyshev_interval
from rta.filter import is_angry
from rta.plot.runs import plot_distances_to_reference

data = Path("~/Projects/rta/rta/data/").expanduser()


A = pd.read_msgpack(data/"annotated_zlib.msg")
A = A.drop(A.columns[0], 1)
U = pd.read_msgpack(data/'unannotated_zlib.msg')
U = U.drop(U.columns[0], 1)

A_cnt = A.shape[0]
U.index = pd.RangeIndex(A_cnt, A_cnt + U.shape[0])
min_runs_per_id = 5 # how many times should peptides appear out of 10 times

## get alignmets
annotated_peptides = A
D, stats, pddra, pepts_per_run = preprocess(A, min_runs_per_id)


# x = 'dt'
def align_and_denoise(A, U, D, x, std_cnt=5):
    """Applicable to any column of A,U,D DataFrames, like rt, dt, mass."""
    x_med = x + '_med'
    x_d = x + '_d'
    xa = x + 'a'
    xa_med = xa + '_med'
    xa_d = xa + '_d'
    xa_dmin, xa_dmax = xa + '_dmin', xa + '_dmax'
    D[x_med] = cond_medians(D[x], D.id)
    runs = np.unique(D.run)
    M = BigModel({r: RollingMedian() for r in runs})
    M.fit(D[x].values, D[x_med].values, D.run.values)
    D[xa] = M(D[x], D.run)
    D[xa_med] = cond_medians(D[xa], D.id)
    D[xa_d] = D[xa_med] - D[xa]
    minmax_xta_d_run = D.groupby('run')[xa_d].apply(robust_chebyshev_interval, std_cnt=std_cnt).apply(pd.Series)
    minmax_xta_d_run.columns = [xa_dmin, xa_dmax]
    A[xa] = M(A[x], A.run)
    A[xa_med] = cond_medians(A[xa], A.id)
    A[xa_d] = A[xa_med] - A[xa]
    A = A.join(minmax_xta_d_run, on='run')
    angry = A.loc[np.logical_or(A[xa_d] < A[xa_dmin], A[xa_d] > A[xa_dmax]),]
    A = A.loc[np.logical_and(A[xa_d] >= A[xa_dmin],  A[xa_d] <= A[xa_dmax]),]
    angry['who'] = xa + '_outlier'
    if not 'who' in U.columns:
        U['who'] = 'no_id'
    U[xa] = M(U[x], U.run)
    U = U.append(angry[U.columns], sort=False)
    return A, U, D, angry, M


def print_NaNs(A, U, D):
    for X in [A, U, D]:
        print(X[['rt', 'rta', 'dt', 'run', 'mass', 'intensity']].isnull().sum().sum())

A, U, D, angry_rt, M_rt = align_and_denoise(A, U, D, 'rt')

W = pd.DataFrame(A[['id', 'charge']].set_index('id').groupby('id').charge.nunique())
W.columns = ['q_cnt']
A = A.join(W, on='id')
multicharged = A.loc[A.q_cnt > 1,] # 9 064
A = A.loc[A.q_cnt == 1,] # 269 291
# print_NaNs(A, U, D)

A, U, D, angry_dt, M_dt = align_and_denoise(A, U, D, 'dt')
# print_NaNs(A, U, D)
# plot_distances_to_reference(A.dta, A.dta_med, A.run, s=1)
# plot_distances_to_reference(A.rta, A.rta_med, A.run, s=1)

A, U, D, angry_mass, M_mass = align_and_denoise(A, U, D, 'mass')

# mass = A.mass
# mass_med = cond_medians(mass, A.id)
# plot_distances_to_reference(mass, mass_med, A.run, s=1)


## saving
A.to_msgpack(data/"A.msg", compress='zlib')
U.to_msgpack(data/"U.msg", compress='zlib')
D.to_msgpack(data/"D.msg", compress='zlib')
angry_rt.to_msgpack(data/"angry_rt.msg", compress='zlib')
angry_dt.to_msgpack(data/"angry_dt.msg", compress='zlib')
angry_mass.to_msgpack(data/"angry_mass.msg", compress='zlib')
multicharged.to_msgpack(data/"multi_q.msg", compress='zlib')
