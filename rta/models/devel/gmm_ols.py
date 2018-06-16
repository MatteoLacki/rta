"""Check the distribution of the error terms from one run of Huber regression.

This way we will ascertain, if only one step of Huber fitting is necessary.
Alternatively, we could refit on the 'signal' data with either L2 or L1 regressors.
Possibly checking for the values of some parameters.
"""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.models.base_model import coef, predict, fitted, coefficients, residuals, res, cv
from rta.models.plot import plot
from rta.models.GMLSQSpline import GMLSQSpline, fit_spline, fit_interlapping_mixtures
from rta.models.RapidGMLSQSpline import RapidGMLSQSpline 
from rta.models.SQSpline import SQSpline, mad_window_filter

data = pd.read_csv("/Users/matteo/Projects/rta/rta/data/one_run_5_folds.csv")
chunks_no = 20


# strategy relying more on pure splines
s_model = SQSpline()
s_model.df_2_data(data, 'rt', 'rt_median_distance')
x, y = s_model.x, s_model.y


%%timeit
s_model = SQSpline()
s_model.df_2_data(data, 'rt', 'rt_median_distance')
s_model.fit(chunks_no=chunks_no)

plot(s_model)
plt.ylim(-3,3) 
plt.show()


# OLD CODE

model = GMLSQSpline()
model.df_2_data(data, 'rt', 'rt_median_distance')
x, y = model.x, model.y

%%timeit
model = GMLSQSpline()
model.fit(x, y, chunks_no, True)

plot(model)
plt.ylim(-3,3) 
plt.show()

# investigating different strategies of fitting to make it faster: premature optimization at its high!
%%timeit
rapid_model = RapidGMLSQSpline()
rapid_model.fit(x, y, chunks_no, True)

# %lprun -f rapid_model.fit rapid_model.fit(model.x, model.y, chunks_no, warn_start=True)

plot(rapid_model)
plt.ylim(-3,3) 
plt.show()
