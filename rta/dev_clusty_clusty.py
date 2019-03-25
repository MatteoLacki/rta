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

data = Path("~/Projects/rta/rta/data").expanduser()
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


from scipy.spatial import cKDTree as kd_tree

runs = np.unique(A.run)

# forest of kd_trees
F = {run: kd_tree(A.loc[A.run == run, ['mass', 'rta', 'dt']]) for run in runs}


# normalizing data in all DataFrames based on D.



u1 = U.loc[U.run == 1,['mass', 'rta', 'dt']]
# this is one time only
kd = cKDTree(u1)

pts = np.array([[0, 0, 0], [2.1, 2.9, 3.3]])
kd.query(pts, k=3)
pt = np.array([[2.1, 2.9, 3.3]])
d, i = kd.query(pt, k=3)
u1.iloc[[8277, 5783, 358199],]

a1 = A.loc[D.run == 1, ['mass', 'rta', 'dt']]

%%time
d, i = kd.query(d1, k=1, p=inf, distance_upper_bound=1)






