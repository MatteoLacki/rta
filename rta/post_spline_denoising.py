%load_ext autoreload
%autoreload 2
import numpy as np
from sklearn import mixture

import plotly.graph_objs as go
from plotly.offline import iplot

from rta.models.base_model import predict, fitted, coef, residuals
from rta.models.huber import huber_spline
from rta.read_in_data import DT as D
from rta.preprocessing import preprocess

DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
hspline = huber_spline(data, formula)
res = residuals(hspline)


plot_data = [go.Histogram(x=res)]
iplot(data, filename='basic histogram')


X = np.asarray(res).reshape((len(res),1))
gmm = mixture.GaussianMixture(n_components=2,
                              covariance_type='full').fit(X)

gmm.covariances_
gmm.weights_

X_new = np.asarray([[.1], [1],[2], [100]])
gmm.predict(X_new)
gmm.predict_proba(X_new)



plt.scatter(hspline.data.rt,
            hspline.data.rt_median_distance,
            c = gmm.predict(X),
            s=.4)
