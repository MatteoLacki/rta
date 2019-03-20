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

data = Path("~/Projects/rta/data/").expanduser()
A = pd.read_msgpack(data/"annotated_all.msg")
U = pd.read_msgpack(data/'unlabelled_all.msg')
min_runs_per_id = 5 # how many times should peptides appear out of 10 times

## get alignmets
D, stats, pddra, pepts_per_run = preprocess(A, min_runs_per_id)
D = D.reset_index()
x = 'rt'

def align_and_denoise(A, U, D, x):
	x_med = x + '_med'
	x_d = x + '_d'
	xa = x + 'a'
	xa_med = xa + '_med'
	xa_d = xa + '_d'
	xa_dmin, xa_dmax = xa + '_dmin', xa + '_dmax'
	D[x_med] = cond_medians(D[x], D.id)
	runs = np.unique(D.run)
	B = BigModel({r: RollingMedian() for r in runs})
	B.fit(D[x].values, D[x_med].values, D.run.values)
	D[xa] = B(D[x], D.run)
	D[xa_med] = cond_medians(D[xa], D.id)
	D[xa_d] = D[xa_med] - D[xa]
	minmax_xta_d_run = D.groupby('run')[xa_d].apply(robust_chebyshev_interval).apply(pd.Series)
	minmax_xta_d_run.columns = [xa_dmin, xa_dmax]
	A[xa] = B(A[x], A.run)
	A[xa_med] = cond_medians(A[xa], A.id)
	A[xa_d] = A[xa_med] - A[xa]
	A = A.join(minmax_xta_d_run, on='run')
	angry = A.loc[np.logical_or(A[xa_d] < A[xa_dmin], A[xa_d] > A[xa_dmax]),]
	A = A.loc[np.logical_and(A[xa_d] >= A[xa_dmin],  A[xa_d] <= A[xa_dmax]),]
	angry = angry.loc[:,['run', 'mass', 'intensity', 'rt', 'dt', 'rta']]
	angry['who'] = xa + '_outlier'
	U = U.loc[:,['run', 'mass', 'intensity', 'rt', 'dt']]
	U[xa] = B(U[x], U.run)
	if not 'who' in U.columns:
		U['who'] = 'no_id'
	U = U.append(angry, sort=False)
	return A, U, D, angry

A, U, D, angry_rt = align_and_denoise(A, U, D, 'rt')

W = pd.DataFrame(A[['id', 'charge']].set_index('id').groupby('id').charge.nunique())
W.columns = ['q_cnt']
A = A.join(W, on='id')
multicharged = A.loc[A.q_cnt > 1,] # 9 064
A = A.loc[A.q_cnt == 1,] # 269 291

A, U, D, angry_dt = align_and_denoise(A, U, D, 'dt')
plot_distances_to_reference(A.dta, A.dta_med, A.run, s=1)
plot_distances_to_reference(A.rta, A.rta_med, A.run, s=1)

## saving
A.to_msgpack(data/"A.msg")
U.to_msgpack(data/"U.msg")
D.to_msgpack(data/"D.msg")
angry_rt.to_msgpack(data/"angry_rt.msg")
angry_dt.to_msgpack(data/"angry_dt.msg")
multicharged.to_msgpack(data/"multi_q.msg")
