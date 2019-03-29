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

from rta.array_operations.dataframe_ops import get_hyperboxes
from rta.reference import cond_medians

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)

AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!
# more than one million free slots

# prepare data for hdbscan
variables = ['rta', 'dta', 'massa']

# almost 4 MLN points...
# AU = pd.concat([ A[variables], U[variables] ], axis=0)
# import hdbscan
# clusterer = hdbscan.HDBSCAN()
# clusterer.fit(AU) # takes FOREVER!!!!
