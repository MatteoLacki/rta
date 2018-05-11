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
from rta.models.plot import plot, plot_curve
from rta.splines.denoising import denoise_and_align
from rta.misc import max_space


# better use natural splines
formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
DF = preprocess(D, min_runs_no = 2)
# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
# WARNING: skipping this for now.
# DF = DF[DF.rt_median_distance != 0]

# All points are 'signal' before denoising: guardians
DF['signal'] = 'signal'
# Division by run number
runs = list(data for _, data in DF.groupby('run'))
%%time
models = list(denoise_and_align(runs, formula))


d = models[0][1]
m = models[0][0]

%matplotlib
plt.scatter(d.dt, d.dt_median_distance, marker = '.')
plt.scatter(d.le_mass, d.le_mass_median_distance, marker = '.')



# resemble all the data sets
DF_2 = pd.concat(d for m, d in models)

import pickle
with open('rta/data/denoised_data.pickle3', 'wb') as h:
    pickle.dump(DF_2, h)
DF_2.to_csv('rta/data/denoised_data.csv', index = False)


%matplotlib
plt.plot(d[d.signal == 'signal'].rt, d[d.signal == 'signal'].rt_aligned)
plt.hist(residuals(m))
%matplotlib
plt.scatter(d.rt,
            d.rt_median_distance,
            c=[{'signal': 'red', 'noise': 'grey'}[s] for s in d.signal])
plot_curve(m, c='blue', linewidth=4)



# Trying out the clustering
from sklearn import cluster
from collections import Counter




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
