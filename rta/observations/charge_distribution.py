"""
What is the distribution of charges?
Dunno, but it is something more concentrated than the Poisson.
"""
%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.special import gammaln

data = Path("~/Projects/rta/rta/data").expanduser()
U = pd.read_msgpack(data/"U.msg")

# distribution of charges.
Uq = U.groupby('charge').size()
mu_est = sum(Uq.index * Uq) / sum(Uq)

n = np.array(Uq.index)
def poiss(n, mu):
	return np.exp(-mu + np.log(mu) * n - gammaln(n+1))

plt.plot(Uq.index, Uq/sum(Uq))
plt.scatter(n, poiss(mu_est, n))
plt.show()