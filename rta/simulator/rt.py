import numpy as np
import numpy.random as npr
import pandas as pd


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


def draw_peptide_rt(size=10000,
                    min_rt=16,
                    max_rt=180,
                    runs_no=10,
                    precision=None,
                    systematic_shifts=None,
                    stack=True):
    """Draw retention times fo peptides.
    
    Args:
        size (int): the number of peptides
        systematic_shifts (functions): the applied functions.
    """
    rt = npr.random(size) * (max_rt - min_rt) + min_rt
    if precision:
        rtp = npr.normal(loc=rt,
                         scale=precision,
                         size=(runs_no, size)).T
        if systematic_shifts:
            assert len(systematic_shifts) == runs_no, "Wrong number of shift functions: is {} should be {}.".format(len(systematic_shifts), runs_no)
            rtps = np.empty(shape=rtp.shape, dtype=rtp.dtype)
            for i in range(runs_no):
                rtps[:,i] = systematic_shifts[i](rtp[:,i])
            if stack:
                return rt, array2df(rtp), array2df(rtps)
            else:
                return rt, rtp, rtps
        if stack:
            return rt, array2df(rtp)
        else:
            return rt, rtp
    return rt


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

def act(F, X):
    """Act with functions in F on the columns of X.

    Create a matrix Y = [F(X1)|F(X2)|...|F(Xd)]

    Args:
        F (tuple of vectorized callables): Functions to apply over the columns of X. Each function should be vectorized,
        X (np.array): A matrix with n rows and d columns.
    """
    n, d = X.shape
    assert len(F) == d, "There are {} functions and {} columns of X.".format(len(F), d)
    Y = np.empty(shape=(n,d), dtype=X.dtype)
    for i in range(d):
        Y[:,i] = F[i](X[:,i])
    return Y

rt = draw_rt()
rts = draw_runs(rt, 3)
shifts = (lambda x: 10 + x * (1 + .01 * np.sin(x/20)),
          lambda x: 7  + x * (1 + .05 * np.cos(x/15)),
          lambda x: 15 + x * (1.01 + .25 * np.sin(x/18)))
rtss = act(shifts, rts)


# automate shift maker
import matplotlib.pyplot as plt

def runif(n, a, b):
    return npr.random(n)*(b-a)+a
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

systematic_shifts = random_diag_sin_shifts(10)
rt, rtp, rtps = draw_peptide_rt(precision=.05,
                                systematic_shifts=systematic_shifts,
                                stack=False)

# x=np.linspace(0,100,10000)
# y=systematic_shifts[0](x)
# plt.plot(x,y)
#z plt.show()
shifts = (lambda x: 10 + x * (1 + .01 * np.sin(x/20)),
          lambda x: 7  + x * (1 + .05 * np.cos(x/15)))

rt, rtp, rtps = draw_peptide_rt(runs_no=2,
                                precision=.5,
                                systematic_shifts=shifts,
                                stack=False)

plt.scatter(rtss[:,0], rtss[:,1])
# plt.scatter(rt, rtps[:,1])
e = rtss.min(), rtss.max()
plt.plot(e, e, color='black')
plt.show()

# adding big noise
rt, rtp, rtps = draw_peptide_rt(runs_no=2,
                                precision=.5,
                                systematic_shifts=shifts,
                                stack=True)
npr.binomial(10000, .01, )
n.shape()
