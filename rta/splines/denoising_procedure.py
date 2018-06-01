%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture, cluster
from collections import Counter

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.models.base_model import predict, fitted, coef, residuals
from rta.models import spline
from rta.models.plot import plot, plot_curve
from rta.splines.denoising import denoise_and_align
from rta.misc import max_space


annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

%%timeit -n 1 -r 1
res = denoise_and_align(annotated,
                        unlabelled,
                        formula,
                        workers_cnt=4)


%%timeit -n 1 -r 1
res = denoise_and_align(annotated,
                        unlabelled,
                        formula,
                        workers_cnt=1)

%%timeit -n 1 -r 1
res = denoise_and_align(annotated,
                        unlabelled,
                        formula,
                        workers_cnt=2)



## data to validate hypothesis about charge reductions and others.
# res.to_csv("~/Projects/retentiontimealignment/Data/rt_denoised.csv",
#            index = False)
model = "Huber"
refit = True

def iter_groups():
    for run_no, a in annotated.groupby('run'):
        u = unlabelled[unlabelled.run == run_no]
        yield a, u, formula, model, refit

a, u, formula, model, refit = next(iter_groups())


%%timeit
model = spline(a, formula)
res = residuals(model).reshape((-1,1))
gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                              covariance_type='full').fit(res)
signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
signal = np.array([signal_idx == i for i in gmm.predict(res)])
if refit:
    # TODO add x-validation to the model here.
    # this will destroy the rest of the code...
    model = spline(a[signal], formula)







# it doesn't make sense for the X-validation to be done globally:
# each run should have it's own procedure.
# it should me a method of the spline class.

from rta.models.sklearn_regressors import sklearnRegression

from patsy import dmatrices


sklearn = sklearnRegression(data=a)
sklearn.fit(formula, 'Huber')
# aha!
# so we can change the formula!
# actually, we have to change the formula, because otherwise we could
# not really pass in the new dimension of the spline.

# cv function prototype

def cv(self):
    for



# Calculate the spaces in different directions for signal points
DF_2_signal = DF_2[DF_2.signal == 'signal']
X = DF_2_signal.groupby('id')
D_stats = pd.DataFrame(dict(runs_no_aligned         = X.rt.count(),
                            rt_aligned_median       = X.rt.median(),
                            rt_aligned_max_space    = X.rt_aligned.aggregate(max_space),
                            pep_mass_max__space     = X.pep_mass.aggregate(max_space),
                            le_mass_max_space       = X.le_mass.aggregate(max_space),
                            dt_max_space            = X.dt.aggregate(max_space)
                            ))
# Maybe instead we could simply calculate the min-max span of feature values.


# this goes towards setting one single value for each cluster.
S = D_stats[D_stats.runs_no_aligned > 2]
S = S.assign(mass2rt = S.le_mass_max_space / S.rt_aligned_max_space,
             dt2rt   = S.dt_max_space      / S.rt_aligned_max_space)

np.percentile(S.mass2rt, q=range(0,110,10))
np.percentile(S.dt2rt, q=range(0,110,10))


DF_2_signal = pd.merge(DF_2_signal,
                       D_stats,
                       left_on='id',
                       right_index=True)

%matplotlib
plt.scatter(X.rt_aligned_median,
            X.rt_aligned_max_space,
            marker = '.')
plt.axes().set_aspect('equal', 'datalim')

diecintili = np.percentile(X.rt_aligned_max_space, q = range(0,110,10))

median_space = diecintili[5]

def brick_metric(x, y):
    return np.linalg.norm(x-y)

DBSCAN = cluster.DBSCAN(eps = median_space,
                        min_samples = 5,
                        metric = brick_metric)
XX = DF_2_signal[['pep_mass', 'rt_aligned', 'dt']]


DBSCAN = cluster.DBSCAN(eps = median_space,
                        min_samples = 5)
XX = DF_2_signal[['rt_aligned']]

dbscan_res = DBSCAN.fit(XX)


%matplotlib
plt.scatter(DF_2_signal.rt,
            DF_2_signal.rt_aligned_max_space,
            marker = '.',
            c = dbscan_res.labels_)
w = Counter(list(dbscan_res.labels_))

del w[-1]
plt.hist(w.values())


dbscan_res.labels_


%matplotlib
plt.scatter(DF_2_signal.rt,
            DF_2_signal.rt_aligned_max_space,
            marker = '.'
            )
plt.axes().set_aspect('equal', 'datalim')
cluster.DBSCAN().


%matplotlib
plt.hist(DF_2_signal.rt_aligned_max_space)


spline(data = )
