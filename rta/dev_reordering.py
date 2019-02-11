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
from rta.align.starlike import StarAligner
from rta.reference import choose_run, choose_most_shared_run, choose_statistical_run

annotated_all, unlabelled_all = big_data()
D, stats, pddra, pepts_per_run = preprocess(annotated_all, 5)
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")
# D.fold will be used in CV, but this will be performed outside A.


X, uX = choose_run(D, 'rt', 1)
X, uX = choose_most_shared_run(D, 'rt', stats)
X, uX = choose_statistical_run(D, 'rt', 'mean')
X, uX = choose_statistical_run(D, 'rt', 'median')




from rta.models.model import Model



Model()
sa = StarAligner()

