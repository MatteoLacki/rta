%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from collections import Counter

from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot


from patsy import demo_data
from patsy import dmatrices, build_design_matrices, dmatrix
from patsy import bs

DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass

K = 6
knots_data = data.rt
# knots_data = np.random.gamma(100, size = 10)

knots = np.percentile(a = knots_data,
                      q = np.linspace(0, 100, 2**K + 1))
len(knots)
knots



spline_basis = bs( x = knots_data,
                   include_intercept = False,
                   knots = knots,
                   degree = 3)


spline_basis = bs( x = knots_data,
                   include_intercept = True,
                   df = 66,
                   degree = 3)

# [np.linalg.cond(bs(x=knots_data, include_intercept = True, df = 2**k, degree = 3)) for k in range(4,10)]
