"""
Some functional programing tools.
"""
import numpy as np

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
