%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# plt.style.use('default')

from plotnine import *

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

def normalize_madly(X, var):
	X[var + "_n"] = X[var] / np.median(np.abs(X[var] - X[var+'_med']))

normalize_madly(A, 'rta')
normalize_madly(A, 'dta')

plt.scatter(A.rta_n, A.dta_n, s=1)
plt.show()

arta = A.loc[np.logical_and(A.rta_n > 9300, A.rta_n < 10000),]
a = arta.loc[np.logical_and(arta.dta_n > 3200, arta.dta_n < 4200),]

plt.scatter(a.rta_n, a.dta_n, s=1, c=a.id)
plt.show()

(ggplot(a, aes(x='rta_n', y='dta_n', color='id')) + 
	geom_point())


(ggplot(A, aes(x='rta', y='dta', xend='rta_med', yend='dta_med')) + 
	geom_segment())


# for each peptide find the bordering rectangle

def get_hyperboxes(X, vars, grouping_var='id'):
	"""Get minimal and maximal values of grouped variables."""
	if not type(vars) == list:
		vars = [vars]
	cols = []
	names = []
	vid = X.groupby(grouping_var)
	for var in vars:
		v = vid[var]
		cols.append(v.min())
		cols.append(v.max())
		names.append(var+"_min")
		names.append(var+"_max")
	B = pd.DataFrame(pd.concat(cols, axis=1))
	B.columns = names
	return B

HB = get_hyperboxes(A, ['rta', 'dta', 'mass'])
HB = HB.reset_index()

(ggplot(HB, aes(xmin='rta_min', xmax='rta_max', ymin='dta_min', ymax='dta_max', group='id')) +
	geom_rect())




def make_peptide_summary(X):
	X['rta_ad'] = np.abs(X.rta - X.rta_med)
	X['dta_ad'] = np.abs(X.dta - X.dta_med)
	Xid = X.groupby('id')
	P = pd.DataFrame(
		pd.concat(
		[Xid.rta_ad.median(),
		 Xid.rta.median(),
		 Xid.dta_ad.median(),
		 Xid.dta.median(),
		 Xid.intensity.median(),
		 Xid.dta.size()],
		axis = 1
		)
	)
	P.columns = ['rta_mad','rta_med','dta_mad','dta_med','intensity','cnt']
	return X, P

A, P = make_peptide_summary(A)



## these show nothing really
# plt.scatter(P.rta_med, P.dta_med, c=P.rta_mad, s=1)
# plt.show()
# plt.scatter(P.dta_med, P.rta_med, c=P.rta_mad, s=1)
# plt.show()

plt.scatter(P.rta_med, P.dta_med, c=P.dta_mad, s=P.rta_mad)
plt.show()

plt.scatter(P.dta_med, P.dta_mad)
plt.show()

plt.scatter(P.dta_med, P.dta_mad, s=1)
plt.show()

# The logs of mad seem to look nicely independent and bell-shaped
plt.scatter(np.log(P.dta_mad), np.log(P.rta_mad), c=np.log(P.intensity), s=1)
plt.show()










plt.scatter(np.log(P.dta_mad), P.rta_mad, s=1)
plt.show()

plt.hexbin(np.log(P.dta_mad), np.log(P.rta_mad))
plt.show()

plt.scatter(np.log(P.dta_mad), np.log(P.rta_mad), s=1)
plt.show()

plt.scatter(np.log(P.dta_mad), np.log(P.dta_med), s=1)
plt.show()

plt.scatter(np.log(P.dta_med), np.log(P.dta_mad), s=1)
plt.show()


(ggplot(P, aes(x='np.log(dta_med)', y='np.log(dta_mad)')) +
	geom_point(size=.2) +
	facet_grid('cnt~.'))

(ggplot(P, aes(x='np.log(dta_med)', y='np.log(dta_mad)', color='cnt', group='cnt')) +
	geom_density_2d())

(ggplot(P, aes(x='np.log(dta_mad)', y='np.log(rta_mad)')) +
	geom_point(size=.2) +
	facet_grid('cnt~.'))


# plt.scatter(D1.dt, D1.rt, s=1)
# plt.show()

# plt.hexbin(D1.dt, D1.rt, gridsize=500)
# plt.show()

plt.scatter(D1.dt, D1.rt, s=1)
plt.show()

plt.hist(D1.dt, bins=1000)
plt.show()


(ggplot(D1, aes(x='dt')) +
	geom_histogram(bins=1000) + 
	facet_grid('charge~.'))
