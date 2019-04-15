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
from collections import Counter
from scipy.spatial import cKDTree as kd_tree
from plotnine import *

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']

Did = D.groupby('id')
HB = pd.concat( [   get_hyperboxes(D, variables),
                    Did[variables].median(),
                    pd.DataFrame(Did.size(), columns=['signal_cnt']),
                    Did.run.agg(frozenset),
                    Did.charge.median(),
                    Did.FWHM.median()     ],
                    axis = 1)
HB = HB[HB.signal_cnt > 5]

plt.hexbin(HB.rta_min, HB.rta_edge)
plt.hexbin(np.log(HB.rta_min), np.log(HB.rta_edge))
plt.show()
plt.hexbin(HB.dta_min, HB.dta_edge)
plt.hexbin(np.log(HB.dta_min), np.log(HB.dta_edge))
plt.show()
(ggplot(HB, aes(x='dta_min', y='dta_edge')) +
    geom_density_2d() +
    facet_wrap('charge'))
# (ggplot(HB, aes(x='FWHM', y='dta_edge')) +
#   geom_density_2d() +
#   facet_wrap('charge'))
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d())
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d() + facet_wrap('signal_cnt'))
(ggplot(HB, aes(x='massa_min', y='massa_edge')) +
    geom_density_2d() +
    facet_wrap('charge'))
(ggplot(HB, aes(x='massa_edge', group='charge', color='charge')) +
    geom_density())
