import numpy as np
import numpy.random as npr
import pandas as pd

from rta.array_operations.functional import act
from rta.stats.random import runif

pi = np.pi

def array2df(x, stack=True):
    x = pd.DataFrame(x)
    x.index = ["p{}".format(i) for i in range(x.shape[0])]
    if stack:
        x = x.stack()
        x.index.names = ['peptide', 'run']
        x = x.reset_index(level=1)
        x.columns = ['run', 'rt']
    return x

def draw_rt(size=10000,
            min_rt=16,
            max_rt=180,
            sort=True):
    """Draw what is considered to be the real retention times.
    
    Args:
        size (int): the number of peptides
        min_rt (float): the minimal retention time.
        max_rt (float): the maximal retention time.
    """
    rt = npr.random(size) * (max_rt - min_rt) + min_rt
    if sort:
        rt.sort()
    return rt

def draw_runs(rt,
              runs_no, 
              precision=.05):
    """Draw the less precise retention times within technical runs.

    Args:
        rt (np.array): the true retention times.
        runs_no (int): the number of technical replicates.
        precision (float): the standard deviation of a gaussian blur of original retention times.
    """
    return npr.normal(loc=rt, scale=precision, size=(runs_no, len(rt))).T


rt = draw_rt()
rts = draw_runs(rt, 3)
shifts = (lambda x: 10 + x * (1 + .01 * np.sin(x/20)),
          lambda x: 7  + x * (1 + .05 * np.cos(x/15)),
          lambda x: 15 + x * (1.01 + .25 * np.sin(x/18)))
rtss = act(shifts, rts)

import matplotlib.pyplot as plt
plt.scatter(rtss[:,0], rtss[:,1])
e = rtss.min(), rtss.max()
plt.plot(e, e, color='black')
plt.show()


# automate shift maker
def random_diag_sin_shifts(n,
                           min_c=0,
                           max_c=10,
                           min_a=.01,
                           max_a=.05,
                           min_f=15,
                           max_f=20):
    """Generate systematics shift sine functions.

    Each one follows a formula:
        f(x) = x + ampl * sin(freq * x)
    """
    C = runif(n, min_c, max_c)
    A = runif(n, min_a, max_a)
    F = runif(n, min_f, max_f)
    return tuple([lambda x: c + x + a*np.sin(2*pi*x/f)
                  for c,a,f in zip(C,A,F)])

rt = draw_rt()
rts = draw_runs(rt, 10)
shifts = random_diag_sin_shifts(10)
rtss = act(shifts, rts)


# add big jumps here.
npr.binomial(10000, .01)