%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
from plotnine import *

from rta.preprocessing import preprocess
from rta.reference import choose_statistical_run
from rta.cv.folds import stratified_grouped_fold
from rta.align.aligner import Aligner
from rta.models.rolling_median import RollingMedian

annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5) # this will be done in MariaDB/Parquet
D, stats = stratified_grouped_fold(D, stats, 10)
X, uX = choose_statistical_run(D, 'rt', 'median')
runs = D.run.unique()
m = {r: RollingMedian() for r in runs} # each run can have its own model
a = Aligner(m)
# this should work also if run is in the index???
a.fit(X)
X['rta'] = a(X)
X = X.reset_index().set_index(['id', 'run'])
D = D.reset_index().set_index(['id', 'run'])
D = D.join(X.rta)
D.reset_index(inplace=True)


D.boxplot(column='rt', by='charge')
plt.show()


(ggplot(D, 
		aes(x='charge',
			y='rt',
			group='charge')) + 
 geom_violin())

(ggplot(D, 
		aes(x='charge',
			y='rta',
			group='charge')) + 
 geom_violin())
