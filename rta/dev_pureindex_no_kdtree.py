"""Checking if the pure indices cannot outcompete the silly kd-tree."""
%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from plotnine import *

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

freevars = ['rta','dta','massa']

Us = U[['run'] + freevars]
As = A[['id','run','charge'] + freevars]

quantile = .99
Aid = A.groupby('id')
As_ranges = Aid[freevars].max() - Aid[freevars].min()
freevars_99_diffs = As_ranges[Aid.size() > 1].quantile(quantile)




As.groupby('run').size().values

%%time
Aid 			= A.groupby('id')
A_mass_box 		= Aid.massa.max() - Aid.massa.min()
massa_diff_999  = np.percentile(A_mass_box[A_mass_box > 0], 99.9)
A_massa = massa = np.sort(A.massa)



L, R = get_intervals_np(A_massa, massa_diff_999)
OC = OpenClosed(L, R)
# OO = OpenOpen(L, R)
A['i'] = OC[A.massa]
Us['i'] = OC[U.massa] # note: any operations on U can be done only once beforehand

Usizes = U.groupby('i').size()

# OK, how to make the same with other variables?
Us_cut = Us[U.i != -1]

# Again, cut everything by masses, then by rta, then by dta.





