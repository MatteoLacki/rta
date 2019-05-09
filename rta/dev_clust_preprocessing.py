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
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

from time import time

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

# freevars = ['rta','dta','massa']
# Did = D.groupby('id')
# HB = pd.concat( [   get_hyperboxes(D, freevars),
#                     Did[freevars].median(),
#                     pd.DataFrame(Did.size(), columns=['signal_cnt']),
#                     Did.run.agg(frozenset), # as usual, the slowest !!!
#                     Did.charge.median(),
#                     Did.FWHM.median()     ],
#                     axis = 1)
# HB = HB[HB.signal_cnt > 5]
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
freevars = ['rta','dta','massa']


Uq = U.groupby('charge').size()
plt.plot(Uq.index, Uq)
plt.show()

grouping = ['id', 'charge']
Dg = D[grouping + freevars].groupby(grouping)
Dagg = Dg.max() - Dg.min()
Dagg.columns = [c+'_range' for c in Dagg.columns]

Dagg_range_centiles = Dagg.groupby('charge').quantile(np.linspace(0,1,11))
Dagg_range_centiles.reset_index(inplace=True)
Dagg_range_centiles.rename(columns={'level_1': 'prob'}, inplace=True)
Dagg_range_centiles = pd.melt(Dagg_range_centiles, id_vars=['charge', 'prob'])

# Trying out different things.
(   ggplot(Dagg_range_centiles, aes(x='value', y='prob', color='variable')) +
	geom_line() +
	facet_grid('charge~.') +
	scale_x_log10() 	)

# there is a lot of single points

Aid 			= A.groupby('id')
A_mass_box 		= Aid.massa.max() - Aid.massa.min()
massa_diff_999  = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)

%%time
L, R = get_intervals_np(A_massa, massa_diff_999)
OC = OpenClosed(L, R)
# OO = OpenOpen(L, R)
A['i'] = OC[A.massa]
U['i'] = OC[U.massa] # note: any operations on U can be done only once beforehand


# 23% of points are not within the projections: not bad, but not extraordinary.
# now, we have to make these intervals a little bit wider
Counter(U.i)[-1] / U.shape[0]
len(np.unique(U.i))-1 # there are a lot of groups
# effectively, iU is the index that replaces part of the tree.

# this query is rather costly, to say the least.
# merge some of the things

U_sizes = U.groupby('i').size()
# merge these points to make selection easier?
%%time
x = list(U.groupby('i'))
# 4.22 secs of retrieving the data the fast way.
# investigate the freaking numpy

plt.scatter(U_sizes.index, U_sizes, s=.1)
plt.show()

