"""
What are the distributions of spreads (max-min) for rta, dta, and massa?
The mass really rocks, compared to other instruments.
"""
%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *
import seaborn as sns
sns.set(style="white")

data = Path("~/Projects/rta/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")

# freevars = ['rta','dta','massa']
freevars = ['rt','dt','mass']
grouping = ['id', 'charge']

Dg = D[grouping + freevars].groupby(grouping)
Dagg = Dg.max() - Dg.min()
Dagg.columns = [c+'_spread' for c in Dagg.columns]

A = pd.read_msgpack(data/"A.msg")
peps10 = A.groupby('id').run.count()
peps10 = peps10[peps10 == 10]
A10 = A[A.id.isin(peps10.index)][['sequence', 'run']+grouping+freevars]
A10g = A10[grouping + freevars].groupby(grouping)
A10agg = A10g.max() - A10g.min()

spread_scaling = A10[freevars].max() - A10[freevars].min()
A10agg = A10agg/spread_scaling
Dagg = A10agg

Dagg_range_centiles = Dagg.groupby('charge').quantile(np.linspace(0,1,11))
Dagg_range_centiles.reset_index(inplace=True)
Dagg_range_centiles.rename(columns={'level_1': 'prob'}, inplace=True)
Dagg_range_centiles = pd.melt(Dagg_range_centiles, id_vars=['charge', 'prob'])

# Trying out different things.
(   ggplot(Dagg_range_centiles, aes(x='value', y='prob', color='variable')) +
	geom_line() +
	facet_grid('charge~.') +
	scale_x_log10() +
	ggtitle("Quantiles of Spreads") +
	theme_light() +
	theme(legend_position="bottom") +
	labs(color='') +
	xlab("normalized spread") +
	ylab("probability"))

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 20)

A10.run = 'run_' + A10.run.astype(str)

A10_rt, A10_dt, A10_mass = [A10.pivot(index='id', columns='run', values=x) for x in ['rt', 'dt', 'mass']]

A10_rt.run1
g = sns.PairGrid(A10_rt, diag_sharey=False)
# g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
# g.map_diag(sns.kdeplot, lw=3)
plt.show()

(  	ggplot(A10_rt) +
	geom_abline(intercept=0, slope=1, color='red') +
	geom_point(aes(x='run_1', y='run_2'), size=.1) +
	coord_fixed() +
	theme_light() +
	xlab('rt in run 1') +
	ylab('rt in run 2')  )

