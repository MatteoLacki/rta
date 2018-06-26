from functools import partial
from itertools import cycle, islice
import numpy as np
from numpy.random import shuffle


def K_folds(N, folds_no):
    K = folds_no
    groups = np.full((N,), 0)
    # produce an array of K numbers repeated 
    N_div_K = N // K
    N_mod_K = N % K
    for i in range(1, K):
        groups[ i * N_div_K : (i+1) * N_div_K ] = i
    if N_mod_K: 
        group_tags = np.arange(K)
        shuffle(group_tags)         # random permutation of groups
        groups[-N_mod_K:] = group_tags[0:N_mod_K] # assigning the remainder
    shuffle(groups)
    return groups


def peptide_stratified_folds(run_cnts, folds_no):
    folds = np.zeros(sum(run_cnts), dtype=np.int8)
    s = e = 0
    for cnt in run_cnts:
        e += cnt
        folds[s:e] = K_folds(cnt, folds_no)
        s = e
    return folds


def iter_tenzer_folds(run_cnts, folds_no=10):
    """Create a Stefan Tenzer's empirical fold iterator.

    Cycles through numbers in set 0,..,folds_no 
    with repetitions for all consecutive strata.

    Args:
        runs_cnts (pandas.core.series.Series): the counts of occurences of peptides in different run groups.
        folds_no (int): the number of folds the data will be divided into.

    Yield:
        a sequence of folds.
    """
    folds = tuple(range(folds_no))
    for cnt in run_cnts:
        for i in islice(cycle(folds), cnt):
            yield i


def shuffled_folds(folds):
    """Shuffle folds.

    Args:
        folds (list of ints): folds numbers.
    Return:
        Infinite sequence of shuffled folds, concatenated one after another.
    """
    while True:
        shuffle(folds)
        for f in folds:
            yield f


def iter_shuffled_tenzer_folds(run_cnts, folds_no=10):
    """Create a randomized Stefan Tenzer's empirical fold iterator.

    Cycles through numbers in set 0,..,folds_no that have been randomly permutated.
    with repetitions for all consecutive strata.

    Args:
        runs_cnts (pandas.core.series.Series): the counts of occurences of peptides in different run groups.
        folds_no (int): the number of folds the data will be divided into.

    Yield:
        a sequence of folds.
    """
    folds = list(range(folds_no))
    for cnt in run_cnts:
        for i in shuffled_folds(folds):
            yield i


def tenzer_folds(run_cnts, folds_no, shuffled=False):
    """Create Stefan Tenzer folds.

    The division into folds takes into account the division
    into different run appearance.
    Within each stratum, we divide data points into subsequent folds
    simply based on their order of appearance in the consider dimension.
    We also repeat the cycles.
    For instance, if retention times were 21.4, 31.5, 53.1, 64.4, 78.2 
    and we wanted 3 folds, then these retention times would be simply mapped to
    folds with numbers 0, 1, 2, 0, 1.
    The 'run_cnts' induce the order of appearance of folds.

    Args:
        runs_cnts (pandas.core.series.Series): the counts of occurences of peptides in different run groups.
        folds_no (int): the number of folds the data will be divided into.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    iter_folds = iter_shuffled_tenzer_folds if shuffled else iter_tenzer_folds
    return np.fromiter(iter_folds(run_cnts, folds_no),
                       count=sum(run_cnts),
                       dtype=np.int8)


randomized_tenzer_folds = partial(tenzer_folds, shuffled=True)