"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)
from rta.read_in_data import big_data
from rta.preprocessing.preprocessing import preprocess


annotated_all, unlabelled_all = big_data()
dp = preprocess(annotated_peptides=annotated_all)


