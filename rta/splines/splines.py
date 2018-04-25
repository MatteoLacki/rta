from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix, build_design_matrices
from patsy import bs, cr, cc
import matplotlib.pyplot as plt
import seaborn as sns

from rta.models.base_model import predict, fitted, coef, residuals
from rta.models.least_squares_splines import least_squares_spline

D = preprocess(D)
for g, data in D.groupby('run'):
    pass



model_l2 = least_squares_spline(data, "rt_median_distance ~ cc(rt, df=20)")
predict(model_l2, rt=[10,20])
model_l2.predict(rt=[10, 25])
model_l2.fitted()

predict(model_l2, rt=[10, 25])
fitted(model_l2)
coef(model_l2)
residuals(model_l2)

model_l2_1 = least_squares_spline(
    data,
    "rt_median_distance ~ bs(rt, df=20, degree=2, include_intercept=True) - 1")

fitted(model_l2_1)

# sr.least_squares()
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ bs(rt, df=40, degree=2, include_intercept=True) - 1")
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ bs(rt, df=100, degree=4, include_intercept=True) - 1")
# sr.plot()
#
# sr.least_squares("rt_median_distance ~ cr(rt, df=20)")
# sr.plot()
#
sr.least_squares("rt_median_distance ~ cc(rt, df=20)")
