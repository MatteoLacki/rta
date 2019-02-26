%load_ext autoreload
%autoreload 2

from collections import Counter
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, dbscan
from sklearn.metrics import confusion_matrix
from sklearn.metrics import v_measure_score # label value independent: can be permuted.

from rta.preprocessing import preprocess
from rta.reference import choose_statistical_run
from rta.cv.folds import stratified_grouped_fold
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedian

unlabelled_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')

D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5) # this will be done in MariaDB/Parquet
D, stats = stratified_grouped_fold(D, stats, 10)



X, uX = choose_statistical_run(D, 'rt', 'median')
runs = D.run.unique()
m = {r: RollingMedian() for r in runs} # each run can have its own model
a = Aligner(m)
# this should work also if run is in the index???
a.fit(X)
X['rta'] = a(X)
X = X.reset_index().set_index(['id', 'run'])
D = D.reset_index().set_index(['id', 'run'])
D = D.join(X.rta)
D.reset_index(inplace=True)

# normalize by the ... what?
D_id = D.groupby('id')
vars = ['rta', 'mass']
spreads = D_id[vars].max() - D_id[vars].min()
spreads['n'] = D_id.size()
# spreads.boxplot(column='rta', by='n')
# plt.show()
# spreads.boxplot(column='mass', by='n')
# plt.show()

# normalizing to a unit mad: this takes so long! Only 17K mads..
mads = D_id[['rta', 'mass']].mad()
mads['n'] = D_id.size()
mad_rta, mad_mass = mads.median().values[0:2]
D['rtan'] = D.rta / mad_rta
D['massn'] = D.mass / mad_mass

# the idea of columns named is still a stupid one..
U = unlabelled_all[['run', 'mass', 'intensity', 'rt', 'dt']].copy()
U['rta'] = a(U.rename(columns={'rt':'x'}))
U['rtan'] = U.rta / mad_rta
U['massn'] = U.mass / mad_mass

G = D[['id', 'run', 'mass', 'intensity', 'rt', 'rta', 'dt', 'rtan', 'massn']].copy()
G = pd.concat([G,U], axis=0, sort=False, ignore_index=True)
# G.to_msgpack('/Users/matteo/Projects/rta/data/for_clust.msg')
# this is data that can be used to test the bloody alignment.

# can DBSCAN retrieve real ids correctly?
# vars = ['rtan', 'massn']
# I = D[vars]

# epses = np.arange(1,101)
# Vs = []
# for eps in epses:
# 	core_samples, clusters = dbscan(X=I, eps=100, min_samples=5, metric='chebyshev')
# 	labels_true = np.array([i for i, r in X.index.values])
# 	V = v_measure_score(labels_true=labels_true, labels_pred=clusters)
# 	Vs.append(V)
# Vs = np.array(Vs)

# plt.plot(epses, Vs)
# plt.show()



from math import inf

def diff_clust(X, delta):
	i = 0
	x_p = inf
	for x in X:
		if x - x_p <= delta:
			i += 1
		yield i
		x_p = x

# 101 ms
clust = list(diff_clust(masses, delta))


x = D.mass.values.copy()

def diff_clust_np(x, quantile=.8, x_unsorted=True):
	"""Cluster 1D input by quantile diff distance.

	Args:
		x (np.array): array.
	"""
	if x_unsorted:
		i = np.argsort(x)
		i = i[i]
		x = np.sort(x)
	D_x = np.diff(x)
	delta = np.quantile(D_x, quantile)
	r = np.full(x.shape, False)
	r[1:] = D_x > delta
	r = np.cumsum(r)
	if x_unsorted:
		return r[i]
	else:
		return r

# 15.5 ms [including all operations]
diff_clust_np(x, .9)

y = np.sort(x)
# 4.72 ms
cl = diff_clust_np(y, .89, False)

Counter(Counter(cl).values())
len(Counter(cl))




# 2.41 sec
G.sort_values('intensity')