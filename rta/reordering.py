%load_ext autoreload
%autoreload 2

import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)

annotated_all, unlabelled_all = big_data()
D, stats, pddra = preprocess(annotated_all, 10)




