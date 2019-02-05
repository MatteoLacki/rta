%load_ext autoreload
%autoreload 2

import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)

annotated_all, unlabelled_all = big_data()
D, stats, pddra = preprocess(annotated_all, 5)


from rta.cv.folds import stratified_grouped_fold

# we have annotated peptides!
D, stats = stratified_grouped_fold(D, stats, 10)
# D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")

# now, elephant in the room: calibration

