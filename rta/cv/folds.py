from itertools import cycle, islice
import numpy as np
from numpy.random import shuffle, choice
import pandas as pd


def K_folds(N, folds_no=10):
    """Simple K-folds with shuffling.

    Args:
        N (int):        number of elements to shuffle.
        folds_no (int): number of folds to generate.
    Returns:
        np.array:       fold assignemt.
    """
    K = folds_no
    groups = np.full((N,), 0)
    # produce an array of K numbers repeated 
    N_div_K = N // K
    N_mod_K = N % K
    for i in range(1, K):
        groups[ i * N_div_K : (i+1) * N_div_K ] = i
    if N_mod_K: 
        # assigning the remainder
        group_tags = np.arange(K)
        shuffle(group_tags) # random permutation of groups
        groups[-N_mod_K:] = group_tags[0:N_mod_K]
    shuffle(groups)
    return groups


def drop(X, col):
    """Drop inplace a given column if it exists."""
    try:
        X.drop(columns=col, inplace=True)
    except KeyError:
        pass


def trivial_K_folds(D, K):
    """This simply adds folds independent of protein-groups or any strata.

    Args:
        D (pd.DataFrame): The data with measurements with colund id.
        K (int): The number of folds.
    Returns:
        tuple with D with additional 'fold' columns.
        If there was one beforehand, then it will be overwritten.
    """
    drop(D, 'fold')
    D['fold'] = K_folds(len(D), K)
    return D


def stratified_grouped_fold(D, stats, K, strata='runs'):
    """Add a folding to the data.

    Args:
        D (pd.DataFrame): The data with measurements with colund id.
        stats (pd.DataFrame): Statistics of data D, containing per peptide info.
        K (int): The number of folds.
        strata (str): String with the name of the column defining the strata in stats.
    Returns:
        tuple with D and stats with additional 'fold' columns.
        If there was one beforehand, then it will be overwritten.
    """
    drop(D, 'fold')
    drop(stats, 'fold')

    # thie gets called on every group
    def f(g):
        g['fold'] = K_folds(len(g), K)
        return g

    # gettin group-stratum specific fold assignments
    stats = stats.groupby(strata).apply(f)

    # passing it to main data
    D = D.join(stats.fold, on='id')

    return D, stats

