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
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed
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
# rq_grouper = RunChargeGrouper(A, U)

Aid 			= A.groupby('id')
A_mass_box 		= Aid.massa.max() - Aid.massa.min()
massa_diff_999  = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)


def get_intervals_np(X, max_diff, sorted=False):
	"""Get left and right ends of the max-diff-net around points X.

	Args:
		X (iterable): points for which we need intervals.
		max_diff (float): half of the max distance between points in the net.
	Returns:
	A tuple of np.arrays with left and right ends of intervals.
	"""
	if not sorted:
		X = np.sort(X)
	dXok = np.diff(X) >= max_diff * 2.0001
	Lidx = np.concatenate((np.array([True]), dXok))
	Ridx = np.concatenate((dXok, np.array([True])))
	return X[Lidx]-max_diff, X[Ridx]+max_diff


%%time
L, R = get_intervals_np(A_massa, massa_diff_999)
OC = OpenClosed(L, R)
# OO = OpenOpen(L, R)
A['i'] = OC[A.massa]
U['i'] = OC[U.massa]

# 23% of points are not within the projections.
# now, we have to make these intervals a little bit wider
Counter(U.i)[-1] / U.shape[0]
len(np.unique(U.i))-1 # there are a lot of groups
# effectively, iU is the index that replaces part of the tree.


A[A.i == 0]

# this query is rather costly, to say the least.
# merge some of the things

A_sizes = []
U_sizes = []
for i in range(len(OC.L)):
	A_sizes.append(A[A.i==i].shape[0])
	U_sizes.append(U[U.i==i].shape[0])


U_sizes = U.groupby('i').size()
# merge these points to make selection easier?



plt.scatter(U_sizes.index, U_sizes)
plt.show()

