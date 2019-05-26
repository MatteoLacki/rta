"""
What are the distributions of spreads (max-min) for rta, dta, and massa?
The mass really rocks, compared to other instruments.
"""
%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *

data = Path("~/Projects/rta/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")

freevars = ['rta','dta','massa']
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
	scale_x_log10() +
	ggtitle("Quantiles of Spreads") +
	theme(legend_position="bottom")	)
