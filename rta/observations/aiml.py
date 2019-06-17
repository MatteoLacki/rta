%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *

from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LinearRegression

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")

A_q2 = A.query("(charge == 2) and (run == 1)")

(ggplot(A_q2) + geom_point(aes("mass", 'dt'), size=1))
LR0 = linregress(A_q2.mass, A_q2.dt)


def ols_res(df, x,  y):
	M = LinearRegression(copy_X=True, fit_intercept=True)
	x, y = df[[x]], df[[y]]
	M.fit(x,y)
	return y - M.predict(x)

A_qr = A.groupby(['charge', 'run'])
errors = A_qr.apply(ols_res, x='mass', y='dt')
A['mass_dt_err'] = errors
D = A[['run', 'mass', 'intensity', 'charge', 'FWHM', 'rt', 'dt', 'LiftOffRT', 'TouchDownRT', 'sequence', 'modification',  'type', 'score', 'mass_dt_err']]

D.to_csv(data/'data_for_dt_prediction.csv', index=False)
D.type.unique()