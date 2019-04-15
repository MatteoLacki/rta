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
from rta.array_operations.dataframe_ops import get_hyperboxes

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']

# How big are the original mass differences?
masses = get_hyperboxes(D, 'mass')
(ggplot(masses, aes(x='mass_edge')) + 
    geom_density() +
    scale_x_log10())

ref_mass = cond_medians(D.massa, D.id)
mass_ppm = (D.massa - ref_mass)/ref_mass * 1e6
mass_ppm = np.abs(mass_ppm[mass_ppm != 0])
(ggplot(pd.DataFrame(mass_ppm), aes(x='massa')) + 
    geom_density() +
    scale_x_log10())
np.quantile(mass_ppm, [.5, .95, .99, .999, 1])

# It seems that most things are below 10 ppm, defined as
# the distane to the median value.