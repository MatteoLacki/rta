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

data = Path("~/Projects/rta/data/").expanduser()
A = pd.read_msgpack(data/"annotated_all.msg")
U = pd.read_msgpack(data/'unlabelled_all.msg')
min_runs_per_id = 5 # how many times should peptides appear out of 10 times

## get alignmets
D, stats, pddra, pepts_per_run = preprocess(A, min_runs_per_id)
D = D.reset_index()
D['rt_me'] = cond_medians(D.rt, D.id)
runs = np.unique(D.run)
B = BigModel({r: RollingMedian() for r in runs})
B.fit(D.rt.values, D.rt_me.values, D.run.values)
D['rta'] = B(D.rt, D.run)

## filtering out the identification that are too far from median
D['rta_me'] = cond_medians(D.rta, D.id)
D['rta_d'] = D.rta_me - D.rta
minmax_rta_d_run = D.groupby('run').rta_d.apply(robust_chebyshev_interval).apply(pd.Series)
minmax_rta_d_run.columns = ['rta_d_min', 'rta_d_max']
A['rta'] = B(A.rt, A.run)
A['rta_me'] = cond_medians(A.rta, A.id)
A['rta_d'] = A.rta_me - A.rta
A = A.join(minmax_rta_d_run, on='run')
angry = A.loc[np.logical_or(A.rta_d < A.rta_d_min,
							A.rta_d > A.rta_d_max),]
A = A.loc[np.logical_and(A.rta_d >= A.rta_d_min,
						 A.rta_d <= A.rta_d_max),]
angry = angry.loc[:,['run', 'mass', 'intensity', 'rt', 'dt', 'rta']]
angry['who'] = 'rt_outlier'
U = U.loc[:,['run', 'mass', 'intensity', 'rt', 'dt']]
U['rta'] = B(U.rt, U.run)
U['who'] = 'no_id'
U = U.append(angry, sort=False)

## getting rid of the dt problem
# dt = A.dt
# dt_me = cond_medians(A.dt, A.id)
# plot_distances_to_reference(dt, dt_me, A.run, s=1)

X = A[['id', 'charge']].set_index('id')
W = X.groupby('id').charge.nunique()
W = pd.DataFrame(W)
W.columns = ['q_cnt']
A = A.join(W, on='id')
multicharged = A.loc[A.q_cnt > 1,] # 9 064
A = A.loc[A.q_cnt == 1,] # 269 291

dt = A.dt
dt_me = cond_medians(A.dt, A.id)
plot_distances_to_reference(dt, dt_me, A.run, s=1)



## saving
A.to_msgpack(data/"A.msg")
U.to_msgpack(data/"U.msg")
angry.to_msgpack(data/"angry.msg")
multicharged.to_msgpack(data/"multi_q.msg")
