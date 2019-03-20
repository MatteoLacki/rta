%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from pathlib import Path

from rta.reference import cond_medians
from rta.filters.angry import is_angry
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedianSpline
from rta.plot.runs import plot_distances_to_reference

data = Path("~/Projects/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

DW = D[['id','run','rt']].pivot(index='id', columns='run', values='rt')
sum(DW.isnull())

from scipy.spatial import KDTree

u1 = U.loc[U.run == 1,['mass', 'rta', 'dt']]
# this is one time only
kd = KDTree(u1)
kd.data
pts = np.array([[0, 0, 0], [2.1, 2.9, 3.3]])
d1 = D.loc[D.run == 1, ['mass', 'rta', 'dt']]

from math import inf
d, i = kd.query(d1, k=10, p=inf, distance_upper_bound=1)

# OK, this is not yet meaningful, but gets there.
