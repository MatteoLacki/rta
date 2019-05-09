"""Checking if the pure indices cannot outcompete the silly kd-tree."""
%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from plotnine import *
from collections import Counter

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

seqs_modifications = A.id.str.split(' ', n=1, expand=True)
seqs_modifications.columns = ['sequence', 'modification']
seqs_modifications.sequence.unique()

mods = Counter()
s = 'Carbamidomethyl+C(7), Carbamidomethyl+C(16), Carbamidomethyl+C(17)'

for s in seqs_modifications.modification.unique():
	for k in s.split(', '):
		mods[k] += 1	
MODS = list(mods.keys())
MODS2 = set([])
for m in MODS:
	MODS2.add(m.split('(')[0])



freevars = ['rta','dta','massa']

# these copies can be avoided, because we will want things to be filtered
Us = U[['run', 'charge'] + freevars].copy()
As = A[['id','run','charge'] + freevars].copy()

quantile = .95
Aid = A.groupby('id')
As_ranges = Aid[freevars].max() - Aid[freevars].min()
freevars_percentile_diffs = As_ranges[Aid.size() > 1].quantile(quantile)
# probably, it doesn't make sense to compare those

# now, we will have to sort few times A : quite fast, as A ain't so big.
def assign_to_interval(col):
	L, R = get_intervals_np(A[col], freevars_percentile_diffs[col])
	OC = OpenClosed(L, R)
	As[col+'_interval'] = OC[A[col]]
	Us[col+'_interval'] = OC[Us[col]]
	return OC

OC = assign_to_interval('massa')
# assign_to_interval('rta')
# assign_to_interval('dta')
# As.groupby('rta_interval').run.size()
# As.groupby('dta_interval').run.size()
# As.groupby('massa_interval').run.size()
## This results in not quite enought clusters: masses nicely divide things, other dimensions don't.

plt.scatter(As.massa, As.rta, s=.1)
plt.show()
plt.scatter(As.massa, As.dta, s=.1, c=As.charge)
plt.show()


%%timeit
quadrant = '(charge == 1) & (massa > 775) & (massa < 780) & (rta > 92) & (rta < 100)'

quadrant = '(charge == 1) & (massa > 776.42) & (massa < 776.5) & (rta > 95) & (rta < 96)'
Us_quad = Us.query(quadrant)
As_quad = As.query(quadrant)
As_quad_id = As_quad.groupby('id')
rects = get_hyperboxes(As_quad, ['rta', 'massa', 'dta'])



( 	ggplot() +
		geom_rect(rects, aes(xmin='massa_min', xmax='massa_max', ymin='rta_min', ymax='rta_max'), alpha=.1, fill='red')+
		geom_text(As_quad, aes('massa', 'rta', label='run'), color='red', size=10) +
		geom_text(Us_quad, aes('massa', 'rta', label='run'), color='blue', size=10) )


( 	ggplot() +
		geom_rect(rects, aes(xmin='massa_min', xmax='massa_max', ymin='dta_min', ymax='dta_max'), alpha=.1, fill='red')+
		geom_text(As_quad, aes('massa', 'dta', label='run'), color='red', size=10) +
		geom_text(Us_quad, aes('massa', 'dta', label='run'), color='blue', size=10) )

( 	ggplot() +
		geom_rect(rects, aes(xmin='rta_min', xmax='rta_max', ymin='dta_min', ymax='dta_max'), alpha=.1, fill='red')+
		geom_text(As_quad, aes('rta', 'dta', label='run'), color='red', size=10) +
		geom_text(Us_quad, aes('rta', 'dta', label='run'), color='blue', size=10) )

