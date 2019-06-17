%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import cKDTree as kd_tree
from plotnine import *
from itertools import islice

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

from time import time

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
# sequenced = A.query('run == 1')
# sequenced[['id', 'rt','dt','mass','charge','intensity']]
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
# unlabelled = U.query('run == 1')
# unlabelled[['rt','dt','mass','charge','intensity']]
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)
# AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
# 100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.max_rows', 5)

# sequenced = A[['sequence', 'run', 'rt','dt','mass','charge','intensity']]
# sequenced.sort_values(['sequence','run'], inplace=True)
# sequenced.iloc[2:12]

plt.hexbin(A.massa, A.rta, bins=300, mincnt=1)
plt.show()

plt.hexbin(A.massa, A.dta, bins=300, mincnt=1)
plt.show()

plt.hexbin(A.rta, A.dta, bins=300, mincnt=1)
plt.show()

freevars=['rta', 'dta', 'massa', 'charge']
cond = '(massa > 1000) and (massa < 1050) and (charge == 2)'
Acond = A[freevars].query(cond)
Ucond = U[freevars].query(cond)

plt.scatter(Ucond.massa, Ucond.dta, c='orange', s=10)
plt.scatter(Acond.massa, Acond.dta, c='blue', s=100, alpha=.1, marker='s')
plt.scatter(Acond.massa, Acond.dta, c='blue', s=10)
plt.show()

plt.scatter(Ucond.massa, Ucond.dta, c='orange', s=10)
plt.scatter(Acond.massa, Acond.dta, c='blue', s=500, alpha=.1, marker='s')
plt.scatter(Acond.massa, Acond.dta, c='blue', s=100, alpha=.2, marker='s')
plt.scatter(Acond.massa, Acond.dta, c='blue', s=10)
plt.show()


import numpy as np   

sequenced = A
sequenced['fold'] = np.random.randint(low=1, high=11, size=(A.shape[0],))
pd.set_option('display.max_rows', 50)
sequenced[['sequence', 'rt','dt','mass','charge','intensity','fold']].head(20)