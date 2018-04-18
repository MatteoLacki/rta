from rta.read_in_data import DT as D
from rta.preprocessing import preprocess
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix, build_design_matrices
from patsy import bs, cr, cc
import matplotlib.pyplot as plt
import seaborn as sns


D = preprocess(D)
for g, data in D.groupby('run'):
    pass


def fit(data, formula):
    outcome, predictors = dmatrices(formula, data)
    coef, residuals, predictors_rank, svals = np.linalg.lstsq(predictors,
                                                              outcome,
                                                              rcond=None)
    coef = coef.ravel()
    return coef, residuals, predictors_rank, svals


coef, res, pred_rank, svals = fit(data, "rt_median_distance ~ rt")
coef, res, pred_rank, svals = fit(data,
    "rt_median_distance ~ bs(rt, df=20, degree=2, include_intercept=True) - 1")

formula = "rt_median_distance ~ bs(rt, df=20, degree=2, include_intercept=True) - 1"
outcome, predictors = dmatrices(formula, data)
coef, residuals, predictors_rank, svals = np.linalg.lstsq(predictors,
                                                          outcome,
                                                          rcond=None)
coef = coef.ravel()
help(np.linspace)
new_data = pd.DataFrame({'rt': np.linspace(min(data.rt),
                                           max(data.rt),
                                           num=1000)})
build_design_matrices()

dmatrix(predictors.design_info, data=new_data, return_type='dataframe')


# Write a class for the SplineRegression:
#   must take in data
#   fit in the model
#   make it possible to fit it
#   plot the data

class SplineRegression(object):
    def __init__(self, data):
        self.data = data
        self.min_rt = min(data.rt)
        self.max_rt = max(data.rt)

    def fit(self, formula):
        self.y, self.X = dmatrices(formula, self.data)
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


sreg = SplineRegression(data)
sreg.fit("rt_median_distance ~ bs(rt, df=20, degree=2, include_intercept=True) - 1")
sreg.plot()

sreg.fit("rt_median_distance ~ bs(rt, df=40, degree=2, include_intercept=True) - 1")
sreg.plot()

sreg.fit("rt_median_distance ~ bs(rt, df=100, degree=4, include_intercept=True) - 1")
sreg.plot()

sreg.fit("rt_median_distance ~ cr(rt, df=100, degree=4, include_intercept=True) - 1")
sreg.plot()




# import statsmodels.api as sm
# import statsmodels.formula.api as smf
#
# # ols = smf.ols("rt_median_distance ~ rt", data=data).fit()
# # ols.summary()






def plot(data, plot_data):
    %matplotlib
    plt.scatter(data.rt, data.rt_median_distance, marker='.')
    plt.plot(plot_data.rt, plot_data.prediction, c='red')



plot(data, plot_data)




dmatrix("rt + np.log(pep_mass)", X)
dmatrix("bs(rt, df=10)", X)


help(np.linalg.lstsq)
