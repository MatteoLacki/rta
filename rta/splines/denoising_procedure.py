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

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

DF = preprocess(D, min_runs_no = 2)

# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
DF = DF[DF.rt_median_distance != 0]

models = { g: spline(data, formula) for g, data in DF.groupby('run')}

g = 1
model = models[1]

for run_no, data in DF.groupby('run'):
    # Fitting the spline
    model = spline(data, formula)

    # Fitting the Gaussian mixture model
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type='full').fit(res)
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    sn = {signal_idx: 'signal', noise_idx: 'noise'}

    gmm.predict(res)



%matplotlib
plot(models[1])




# %matplotlib
# divmod(6,5)
# for g, model in models.items():
#     div, mod = divmod(g, 5)
#     plt.subplot(2, div + 1, mod)
#     plot(model)
#
# plt.subplot(2, 2, 2)
# plot(models[2])
