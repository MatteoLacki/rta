%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from patsy import bs

from rta.models.huber import huber_spline
from rta.read_in_data import DT as D
from rta.preprocessing import preprocess

DF = preprocess(D, min_runs_no = 2)
for g, data in DF.groupby('run'):
    pass

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"

from patsy import ModelDesc
ModelDesc.from_formula(formula)




# Huber regression
hspline = huber_spline(data, formula)

X = bs( data.rt,
        df=10,
        knots=None,
        degree=3,
        include_intercept=True,
        lower_bound=0,
        upper_bound=200 )




# sd = np.sqrt(gmm_ols.covariances[:,0])
# plt.plot(range(len(sd)), sd)
# plt.show()


# comparing the two fits
plt.subplot(2, 1, 1)
gmm_ols = GMM_OLS()
gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no)
plot(gmm_ols)
plt.ylim(-3,3) 

plt.subplot(2, 1, 2)
gmm_ols = GMM_OLS()
gmm_ols.fit(formula, data, data_sorted=True, chunks_no=chunks_no, weighted=True)
plot(gmm_ols)

plt.ylim(-3,3) 
plt.show()


# Now: we have to sync the generation of spline with the percentiles of the control variable
# 
# should the formula be split?
# we need to call bs function directly because of the need to change df?
# maybe as a transform method???

# how patsy things are generated in the first place?
from patsy import bs, cc, cr


plt.title("B-spline basis example (degree=3)");
x = np.linspace(0., 1., 100)
y1= dmatrix(bs(x, df=6, degree=3, include_intercept=True))
y = dmatrix("bs(x, df=6, degree=1, include_intercept=True) - 1", {"x": x})



b = np.array([1.3, 0.6, 0.9, 0.4, 1.6, 0.7])
plt.plot(x, y*b);
plt.plot(x, np.dot(y, b), color='white', linewidth=3);
plt.show()

(y*b).shape

y1.design_info
y.design_info



control = gmm_ols.control

control[list(percentiles(len(control), chunks_no))]
list(percentiles(6, 3))



