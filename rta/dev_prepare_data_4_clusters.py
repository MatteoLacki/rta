%load_ext autoreload
%autoreload 2

from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from rta.read.csvs import big_data
from rta.reference import cond_medians
from rta.preprocessing import preprocess
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedian
from rta.math.stats import med_mad, robust_chebyshev_interval
from rta.filter import is_angry
from rta.plot.runs import plot_distances_to_reference
from rta.array_operations.dataframe_ops import normalize

data = Path("~/Projects/rta/rta/data/").expanduser()

A = pd.read_msgpack(data/"annotated_zlib.msg")
A = A.drop(A.columns[0], 1)
U = pd.read_msgpack(data/'unannotated_zlib.msg')
U = U.drop(U.columns[0], 1)

A_cnt = A.shape[0]
U.index = pd.RangeIndex(A_cnt, A_cnt + U.shape[0])
min_runs_per_id = 5 # how many times should peptides appear out of 10 times

annotated_peptides = A
D, stats, pddra, pepts_per_run = preprocess(A, min_runs_per_id)
# x = 'dt'

def align(A, U, D, x):
    """Align A and U based on D.

    Args:
        A (pd.DataFrame): All identified signals.
        U (pd.DataFram): All unidentified signals.
        D (pd.DataFrame): Subset of identified signals that are cleaner.
        x (str): Name of the column that needs to be aligned.
    """
    D[x+'_med'] = cond_medians(D[x], D.id)
    runs = np.unique(D.run)
    M = BigModel({r: RollingMedian() for r in runs})
    M.fit(D[x].values, D[x+'_med'].values, D.run.values)
    D[x+'a'] = M(D[x], D.run)
    D[x+'a_med'] = cond_medians(D[x+'a'], D.id)
    D[x+'a_d'] = D[x+'a_med'] - D[x+'a']
    A[x+'a'] = M(A[x], A.run)
    A[x+'a_med'] = cond_medians(A[x+'a'], A.id)
    A[x+'a_d'] = A[x+'a_med'] - A[x+'a']
    U[x+'a'] = M(U[x], U.run)
    return A, U, D, M

def get_chebs(X, xa, std_cnt=5):
    chebs = D.groupby('run')[xa+'_d'].apply(robust_chebyshev_interval,
                                            std_cnt=std_cnt).apply(pd.Series)
    chebs.columns = [xa+'_dmin', xa+'_dmax']
    return chebs

def split(X, crit):
    return X.loc[crit], X.loc[~crit]

# X = A
def apply_cheb(X, chebs, xa):
    X = X.join(chebs, on='run')
    w = np.logical_and(X[xa+'_d'] >= X[xa+'_dmin'], X[xa+'_d'] <= X[xa+'_dmax'])
    X, X_noise = split(X, w)
    X_noise['who'] = xa + '_outlier'
    return X, X_noise

# xa = 'rta'
def denoise_cheb(A, U, D, xa, std_cnt=5):
    """Denoise the aligned data by applying a robust Chebyshev interval selection.

    Args:
        A (pd.DataFrame): All identified signals.
        U (pd.DataFram): All unidentified signals.
        D (pd.DataFrame): Subset of identified signals that are cleaner.
    """
    chebs = get_chebs(D, xa, std_cnt)
    A, A_noise = apply_cheb(A, chebs, xa)
    D, D_noise = apply_cheb(D, chebs, xa)
    if not 'who' in U.columns:
        U['who'] = 'no_id'
    U = U.append(A_noise[U.columns], sort=False)
    U = U.append(D_noise[U.columns], sort=False)
    return A, A_noise, D, D_noise, U 

def print_NaNs(A, U, D):
    for X in [A, U, D]:
        print(X[['rt', 'rta', 'dt', 'run', 'mass', 'intensity']].isnull().sum().sum())

A, U, D, M_rt = align(A, U, D, 'rt')
A, A_noise_rt, D, D_noise_rt, U = denoise_cheb(A, U, D, 'rta')
# OK, and now, reapply the reallignment

W = pd.DataFrame(A[['id', 'charge']].set_index('id').groupby('id').charge.nunique())
W.columns = ['q_cnt']
A = A.join(W, on='id')
multicharged = A.loc[A.q_cnt > 1,] # 9 064
A = A.loc[A.q_cnt == 1,] # 269 291
# print_NaNs(A, U, D)

A, U, D, M_dt = align(A, U, D, 'dt')
A, A_noise_dt, D, D_noise_dt, U = denoise_cheb(A, U, D, 'dta')

A, U, D, M_mass = align(A, U, D, 'mass')
A, A_noise_m, D, D_noise_m, U = denoise_cheb(A, U, D, 'massa')


A, U, D, M_rt = align(A, U, D, 'rt')
A, U, D, M_dt = align(A, U, D, 'dt')
A, U, D, M_mass = align(A, U, D, 'mass')
# print_NaNs(A, U, D)
# plot_distances_to_reference(A.dta, A.dta_med, A.run, s=1)
# plot_distances_to_reference(A.rta, A.rta_med, A.run, s=1)
# plot_distances_to_reference(A.massa, A.massa_med, A.run, s=1)

# OK, so now, we should perform all that again.

def get_normalization(X, var):
    return np.median(np.abs(X[var] - X[var+'_med']))

for var in ['rta', 'dta', 'massa']:
    nor = get_normalization(A, var)
    for X in [A, U, D]:
        normalize(X, var, nor)


## saving
A.to_msgpack(data/"A.msg", compress='zlib')
U.to_msgpack(data/"U.msg", compress='zlib')
D.to_msgpack(data/"D.msg", compress='zlib')
A_noise_rt.to_msgpack(data/"A_noise_rt.msg", compress='zlib')
D_noise_rt.to_msgpack(data/"D_noise_rt.msg", compress='zlib')
A_noise_dt.to_msgpack(data/"A_noise_dt.msg", compress='zlib')
D_noise_dt.to_msgpack(data/"D_noise_dt.msg", compress='zlib')
A_noise_m.to_msgpack(data/"A_noise_m.msg", compress='zlib')
D_noise_m.to_msgpack(data/"D_noise_m.msg", compress='zlib')
multicharged.to_msgpack(data/"multi_q.msg", compress='zlib')

