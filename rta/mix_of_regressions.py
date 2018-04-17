from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix

D = preprocess(D)
for g, X in D.groupby('run'):
    pass

X.columns
outcome, predictors = dmatrices("rt_median_distance ~ rt + pep_mass", X)

dmatrix("rt_median_distance ~ rt + pep_mass", X)
