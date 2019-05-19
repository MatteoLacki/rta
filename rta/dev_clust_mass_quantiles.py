"""
Are mass division good for speeding up cKD-Tree by pre-branching?

"""
%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *
from scipy.spatial import cKDTree as kd_tree

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

freevars = ['rta','dta','massa']
grouping = ['id', 'charge']

Aid 			= A.groupby('id')
A_mass_box 		= Aid.massa.max() - Aid.massa.min()
massa_diff_999  = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)

K = 51 # Number of clusters
bins = np.linspace(0,1,K)

%%time
x = pd.qcut(A.massa, bins)

%%time
K = 51 # Number of clusters
bins = np.linspace(0,1,K)
x = pd.qcut(U.massa, bins)

%%time
x = pd.cut(U.massa, bins)

%%time
bins = np.quantile(U.massa, np.linspace(0,1,K))
x = np.digitize(U.massa, bins)

