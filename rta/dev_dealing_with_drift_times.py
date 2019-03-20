%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from plotnine import *

from rta.dev_get_data import get_data
from rta.reference import cond_medians
from rta.plot.runs import plot_distances_to_reference
from rta.filters.angry import is_angry

D, U, run, mass, q, rt, rta, dt, ids = get_data()
dt_me = cond_medians(dt, ids)
# plot_distances_to_reference(dt, dt_me, run, s=1)

D_dt = dt_me - dt
angry_dt = is_angry(D_dt)

# plt.scatter(dt, D_dt, s=1, c=angry_dt)
# plt.show()

X = D[['id', 'charge']].set_index('id')
W = X.groupby('id').charge.nunique()
W = pd.DataFrame(W)
W.columns = ['q_cnt']
D = D.set_index('id')
D = pd.concat([D,W], join='inner', axis=1)
# plt.scatter(dt, dt_me-dt, s=1, c=D.q_cnt)
# plt.show()
q_cnt = D.q_cnt.values

plot_distances_to_reference(dt[q_cnt==1],
                            dt_me[q_cnt==1],
                            run[q_cnt==1],
                            s=1)

# plt.scatter(dt[q_cnt==1], dt_me[q_cnt==1]-dt[q_cnt==1], s=1)
# plt.show()
sum(q_cnt > 1) # annotated 6602 peptides
# points are diagonal, because we plot distances to medians,
# and for all the runs

W_same_charge_r = W[W.q_cnt == 1].sample(100)
D_same_charge_r = D.loc[W_same_charge_r.index,]
dt_r = D_same_charge_r.dt
dt_r_me = cond_medians(dt_r, D_same_charge_r.index)
D_same_charge_r['dt_me'] = dt_r_me
D_same_charge_r['D_dt'] = dt_r_me - dt_r
# plt.scatter(dt_r, dt_r_me - dt_r, s=1)
# plt.show()


