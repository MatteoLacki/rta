from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from patsy import bs, cr, cc
import matplotlib.pyplot as plt


D = preprocess(D)
for g, data in D.groupby('run'):
    pass

outcome, predictors = dmatrices("rt_median_distance ~ rt", data)
coefs = np.linalg.lstsq(predictors, outcome, rcond=None)[0].ravel()

outcome, predictors = dmatrices(
    "rt_median_distance ~ bs(rt, df=10, degree=3, include_intercept=True) - 1",
    data)
sol, residuals, predictors_rank, s = np.linalg.lstsq(predictors,
                                                     outcome,
                                                     rcond=None)

fit = np.dot(predictors, sol)
fit.shape


outcome, predictors = dmatrices(
    "rt_median_distance ~ bs(rt, df=10, degree=3, include_intercept=True) - 1",
    X)



dmatrix("rt + np.log(pep_mass)", X)
dmatrix("bs(rt, df=10)", X)


help(np.linalg.lstsq)
/home/matteo/Projects/retentiontimealignment/Py/rta/py3_6/lib/python3.6/site-packages/ipykernel_launcher.py:1:
FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
  """Entry point for launching an IPython kernel.
