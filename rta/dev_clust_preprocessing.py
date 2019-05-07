%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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


freevars = ['rta','dta','massa']
Did = D.groupby('id')
HB = pd.concat( [   get_hyperboxes(D, freevars),
                    Did[freevars].median(),
                    pd.DataFrame(Did.size(), columns=['signal_cnt']),
                    Did.run.agg(frozenset), # as usual, the slowest !!!
                    Did.charge.median(),
                    Did.FWHM.median()     ],
                    axis = 1)
HB = HB[HB.signal_cnt > 5]
# (ggplot(HB, aes(x='massa_min', y='massa_edge')) + 
#     geom_density_2d() + 
#     coord_fixed() + 
#     scale_y_log10() +
#     scale_x_log10() +
#     facet_grid('~charge'))

# getting back to the idea of mass divisions.
# (ggplot(A, aes(x='massa')) + 
#     geom_histogram(bins=3000))
rq_grouper = RunChargeGrouper(A, U)

%%time
Aid = A.groupby('id')
A_mass_box = Aid.massa.max() - Aid.massa.min()
massa_diff_999 = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)

def get_intervals(X, max_diff):
	x = iter(X)
	L = x_ = next(x)
	for _x in x:
		if _x - x_ >= max_diff:
			yield (L, _x)
			L = _x
		x_ = _x

def get_intervals_np(X, max_diff):
	dXok = np.diff(X) >= max_diff
	Lidx = np.concatenate((np.array([True]), dXok))
	Ridx = np.concatenate((dXok, np.array([True])))
	L, R = X[Lidx], X[Ridx]
	return L[L < R], R[L < R] # kill lonely lonelers

# maybe make the intervals slightly bigger?
# but definately don't make it into a division.



L, R = get_intervals_np(A_massa, massa_diff_999)

x = A.massa.values

L = np.array([1, 10, 100])
R = np.array([4, 14, 105])

x = 1
np.searchsorted(L, x)


np.searchsorted()


%%time
OI = SpectralIntervals(L, R)
w = OI[A.massa]
y = OI[U.massa]






plt.step(p, np.percentile(massa_edge, p))
plt.show()

get_hyperboxes(A, 'mass')
Arq = A.groupby(['run', 'charge'])
A12 = Arq.get_group((1,2))

mass = np.sort(A12.mass)
mass = np.sort(A.mass)
np.percentile(np.diff(np.sort(A_mass)), [50, 90, 99, 100])

# plt.step(mass[:-1], np.cumsum(np.diff(mass)))
plt.step(mass[:-1], np.diff(mass))
plt.show()

plt.scatter(mass[:-1], np.diff(mass), s=.5)
plt.show()