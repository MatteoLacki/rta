%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from patsy import bs

from rta.models.huber import huber_spline
from rta.read_in_data import DT as D
from rta.preprocessing import preprocess

DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

from patsy import ModelDesc
ModelDesc.from_formula(formula)




# Huber regression
hspline = huber_spline(data, formula)

X = bs( data.rt,
        df=10,
        knots=None,
        degree=3,
        include_intercept=True,
        lower_bound=0,
        upper_bound=200 )
