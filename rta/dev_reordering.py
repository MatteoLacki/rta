%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from rta.plot.runs import plot_distances_to_reference
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run
from rta.reference import stat_reference
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedianSpline

# %%time
unlabelled_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')
# 2.29 s

# Preprocessing could be done more efficiently elsewhere?
# %%time
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
# 6.32 s
runs = D.run.unique()

# %%time
D, stats = stratified_grouped_fold(D, stats, 10)
# 1.43 s

# %%time
X, uX = choose_statistical_run(D, 'rt', 'median')
# 250 ms

rmi = RollingMedianSpline()
m = {r: RollingMedianSpline() for r in runs} # each run can have its own model
a = Aligner(m)

# %%time
a.fit(X)
# 947 ms

# a.plot(s=1)
a(X)




