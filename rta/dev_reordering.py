%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)

from rta.read.csvs import big_data
from rta.preprocessing import preprocess
from rta.cv.folds import stratified_grouped_fold
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run

annotated_all, unlabelled_all = big_data()
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")
# D.fold will be used in CV, but this will be performed outside A.

# X, uX = choose_run(D, 'rt', 1)
# X, uX = choose_most_shared_run(D, 'rt', stats)
# X, uX = choose_statistical_run(D, 'rt', 'mean')
X, uX = choose_statistical_run(D, 'rt', 'median')


from rta.models.model import Model
from rta.align.starlike import StarAligner


# need some real model by now.
from rta.models.rolling_median import RollingMedian

X1 = X.loc[X.run == 1,]
x = X1.x.values
y = X1.y.values - X1.x.values
rmi = RollingMedian()


runs = D.run.unique()
m = {r: RollingMedian() for r in runs}
sa = StarAligner(m)
sa.m

#TODO: now it's time to play with the two aligning strategies.
	# eliminate the requirement to sort anything.
	# the outputs of the method should be writen down to X and returned.