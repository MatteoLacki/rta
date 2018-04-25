from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix, build_design_matrices
from patsy import bs, cr, cc
import matplotlib.pyplot as plt
import seaborn as sns
from rta.LGM.src.linear_regression_mixtures import LinearRegressionsMixture as LRM

D = preprocess(D)
for g, data in D.groupby('run'):
    pass


class SplineRegression(object):
    def __init__(self, data):
        self.data = data
        self.min_rt = min(data.rt)
        self.max_rt = max(data.rt)

    def dmatrices(self, formula):
        self.y, self.X = dmatrices(formula, self.data)

    def least_squares(self, formula):
        self.dmatrices(formula)
        b, self.res, self.pred_rank, self.svals = np.linalg.lstsq(self.X,
                                                                  self.y,
                                                                  None)
        self.coef = b.ravel()

    def predict(self, x=None):
        if not x:
            x = pd.DataFrame({'rt': np.linspace(self.min_rt,
                                                self.max_rt,
                                                num = 1000)})
        filtered_x = dmatrix(self.X.design_info, data = x)
        prediction = pd.DataFrame({'rt': x.rt,
                                   'prediction': np.dot(filtered_x,
                                                        self.coef)})
        prediction.sort_values(by=('rt'))
        return prediction

    def plot(self, x=None):
        prediction = self.predict(x)
        %matplotlib
        plt.style.use('dark_background')
        plt.scatter(self.data.rt,
                    self.data.rt_median_distance,
                    s=.4)
        plt.plot(prediction.rt, prediction.prediction, c='red')


sr = SplineRegression(data)
# sr.least_squares("rt_median_distance ~ bs(rt, df=20, degree=2, include_intercept=True) - 1")
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
sr.plot()


sr.dmatrices("rt_median_distance ~ bs(rt, df=20, include_intercept=True) - 1")
np.asarray(sr.X)

# Model parameters
K = 2
epsilon = 1e-4
lam = 0.1
iterations = 50
random_restarts = 20

model = LRM(sr.X, sr.y, K=K)
model.train(epsilon=epsilon,
            lam=lam,
            iterations=iterations,
            random_restarts=random_restarts,
            verbose=True)

print(model)

from rta.regression_mixtures.linear_regression_mixtures import fit_with_restarts
results = fit_with_restarts
