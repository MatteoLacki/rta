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


# Patsy tests ##################################
from patsy import demo_data
from patsy import dmatrices, build_design_matrices, dmatrix

data = demo_data('x', 'y', 'a')
X = np.column_stack(([1] * len(data["y"]), data["x"]))
dmatrices((data['y'], X), data=None)
dmatrices('y ~ x', data=data)
help(dmatrix)

build_design_matrices((data['y'], X), data=None)

################################################


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

DF = preprocess(D, min_runs_no = 2)

# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
DF = DF[DF.rt_median_distance != 0]
DF['signal'] = 'signal' # guardian
runs = list(data for _, data in DF.groupby('run'))

data = runs[0]
data.loc[data.run == 1, 'pep_mass'] = 10

T = pd.DataFrame(dict(a=['a','a','b'], x=[2.1, 1.0, 9.0]))
T.loc[T.a == 'a', 'x'] = 10



def denoise(runs, filter_noise=False):
    """Remove noise in grouped data.

    Updates the 'signal' column in the original data chunks.
    """
    for data in runs:
        # Fitting the spline
        model = spline(data, formula)
        # Fitting the Gaussian mixture model
        res = residuals(model).reshape((-1,1))
        gmm = mixture.GaussianMixture(n_components=2,
                                      covariance_type='full').fit(res)
        signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
        sn = {signal_idx: 'signal', noise_idx: 'noise'}
        # appending to the data set
        data.signal = [ sn[i] for i in gmm.predict(res)]
        yield data

%%time
runs1 = list(denoise(runs))

Counter(s for run in runs1 for s in run.signal)





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
