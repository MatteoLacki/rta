%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('dark_background')


data = Path("~/Projects/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")


AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!

# plt.hexbin(A.mass, A.dt, gridsize=200)
# plt.show()
# plt.hexbin(A.mass, A.rt, gridsize=200)
# plt.show()
# plt.hexbin(A.dt, A.rt, gridsize=200)
# plt.show()


from scipy.spatial import KDTree, cKDTree


u1 = U.loc[U.run == 1,['mass', 'rta', 'dt']]
# this is one time only
kd = KDTree(u1)
kd.data
pts = np.array([[0, 0, 0], [2.1, 2.9, 3.3]])
d1 = D.loc[D.run == 1, ['mass', 'rta', 'dt']]

from math import inf
d, i = kd.query(d1, k=10, p=inf, distance_upper_bound=1)

# OK, this is not yet meaningful, but gets there.


