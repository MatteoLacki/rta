import numpy as np
from numpy.random import shuffle


def grouped_K_folds(x, K):
    """Get K-folds that respects the inner groupings.

    Divide the elements of "x" into "K" folds.
    Some elements of "x" might repeat.
    Repeating elements must be assigned together to one of the folds.

    Args:
        x (iterable) : x to divide
        K (int) : number of folds

    Return:
        out (dict) : mapping between unique x and their 
    """
    ids = np.unique(x)
    N = len(ids)
    N_div_K = N // K
    N_mod_K = N % K
    groups = np.full((N,), 0)
    for i in range(1, K):
        groups[ i * N_div_K : (i+1) * N_div_K ] = i
    if N_mod_K: 
        # we decide at random (sampling without repetition)
        # to which groups we should assign the remaining free indices.
        group_tags = list(range(K))
        shuffle(group_tags)
        groups[-N_mod_K:] = group_tags[0:N_mod_K]
    shuffle(groups)
    id_2_group = dict(zip(ids, groups))
    return np.array([id_2_group[x] for x in x], dtype=np.int8) 