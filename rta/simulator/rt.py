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

# draw_peptide_rt(precision=.05)
# systematic_shift = lambda x: x**2
# systematic_shifts = tuple([systematic_shift for _ in range(10)])
# draw_peptide_rt(precision=.05,
#                 systematic_shifts=systematic_shifts,
#                 stack=False)
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

plt.scatter(rtps[:,0], rtps[:,1])
# plt.scatter(rt, rtps[:,1])
e = rtps.min(), rtps.max()
plt.plot(e, e, color='black')
plt.show()
