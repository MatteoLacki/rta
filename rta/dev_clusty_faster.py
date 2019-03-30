%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from collections import Counter
from scipy.spatial import cKDTree as kd_tree
from plotnine import *

from rta.array_operations.dataframe_ops import get_hyperboxes

data = Path("~/Projects/rta/rta/data").expanduser()
A,D,U = [pd.read_msgpack(data/(x+".msg")) for x in ("A", "D", "U")]
vars_ = ['massa', 'rta', 'dta']
runs = np.unique(A.run)
A['signal2medoid_d'] = np.abs(A[[v+"_d" for v in vars_]]).max(axis=1)
# AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
# 100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# more than one million free slots
Aid = A.groupby('id')
A_agg = pd.concat(	[Aid[vars_].median(),
					 pd.DataFrame(Aid.size(), columns=['signal_cnt']),
					 Aid.run.agg(frozenset)],
					 axis = 1) # peptide-medoids
A[A.id == "AAEEGAK NA"]
# shit, after filtering noise, I did not recalculate the stats
# and I should, because otherwise I have single clusters that have
# non-zero distance to the median, which is absurd and also compromises
# the data.

# Fix that.

#this takes some time
F = {run: kd_tree(U.loc[U.run == run, vars_]) for run in runs}
n_jobs = 1 # this does not scale good.

%%time
nn = {}
for r in runs:
	peps_not_in_r = A_agg.loc[[r not in R for R in A_agg.run], vars_]
	nn[r] = F[r].query(peps_not_in_r, p=inf, k=1, n_jobs=n_jobs)
# OK: what now? 
## filter wrong things
## look at the distribution of the infinite norm on the real clusters 
## and choose a cutoff
A['sup_d'] = np.abs(A[[v+'_d' for v in vars_]]).max(axis=1)
A_agg['sup_d'] = A.groupby('id').sup_d.max()

(ggplot(A_agg[A_agg.sup_d > 0], aes(x='sup_d')) +
	geom_histogram(bins=100) +
	facet_wrap('signal_cnt'))

A_agg[A_agg.signal_cnt == 1][A_agg.sup_d > 0]