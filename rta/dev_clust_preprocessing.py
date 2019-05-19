"""
Are mass division good for speeding up cKD-Tree by pre-branching?

"""
%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *
from scipy.spatial import cKDTree as kd_tree

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

freevars = ['rta','dta','massa']
grouping = ['id', 'charge']

Aid 			= A.groupby('id')
A_mass_box 		= Aid.massa.max() - Aid.massa.min()
massa_diff_999  = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)

# %%time
L, R = get_intervals_np(A_massa, massa_diff_999)
OC = OpenClosed(L, R)
# OO = OpenOpen(L, R)
A['i'] = OC[A.massa]
U['i'] = OC[U.massa] # note: any operations on U can be done only once beforehand
# 23% of points are not within the projections: not bad, but not extraordinary.
# now, we have to make these intervals a little bit wider
Counter(U.i)[-1] / U.shape[0]
len(np.unique(U.i))-1 # there are a lot of groups
U_sizes = U.groupby('i').size()
# plt.scatter(U_sizes.index, U_sizes, s=.1)
# plt.show()

K = 51 # Number of clusters
# how to cluster the mass bins???
# -> try to divide equally? simple consecutive clusters (alternative: each Kth clustered together)

# this assumes that index is sorted.
# is that always true? Need a check!

bins = np.linspace(0,1,K)

%%time
x = pd.qcut(
	U.i,
	bins)

%%time
uu = np.cumsum(U_sizes.values[1:])
uu = uu/uu[-1]
bins = np.linspace(0,1,K)
bins[-1] = inf

Uu = dict(	zip(U_sizes.index[1:], 
			np.digitize(uu, bins))	)
Uu[-1] = -1
U['clustered_i'] = U.i.map(Uu) # this is super-fast!
# not all things here have equivalent A index:
# redo calculations for A.i entirely!!!!

# Probably this could be done more directly.
Ai = np.unique(A.i)
aa = np.cumsum(Ai)
aa = aa/aa[-1]
Aa = dict(zip(Ai, np.digitize(aa, bins)))
A['clustered_i'] = A.i.map(Aa) # this is super-fast!


h = U.groupby('clustered_i').size()
# plt.scatter(h.index, h, s=.1)
# plt.show()

# which things get
A.i

