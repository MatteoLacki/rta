%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot


DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
model = spline(data, formula)
%matplotlib
plot(model)
