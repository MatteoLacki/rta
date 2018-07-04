from itertools import cycle, islice
import numpy as np
from numpy.random import shuffle, choice
import pandas as pd


def K_folds(N, folds_no=10):
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



def peptide_stratified_folds(run_cnts, folds_no=10):
    folds = np.zeros(sum(run_cnts), dtype=np.int8)
    s = e = 0
    for cnt in run_cnts:
        e += cnt
        folds[s:e] = K_folds(cnt, folds_no)
        s = e
    return folds



def shuffled_cycle(folds):
    """Shuffled cycles.

    Args:
        folds (list of ints): folds numbers.
    Return:
        Infinite sequence of shuffled folds, concatenated one after another.
    """
    while True:
        shuffle(folds)
        for f in folds:
            yield f



def iter_stratified_folds(strata_cnts, folds_no=10, shuffle=False):
    """Iterate over assignments to different strata,

    Cycles through numbers in set 0,..,folds_no 
    with repetitions for all consecutive strata.

    Args:
        strata_cnts (iterable): counts of elements in subsequent strata.
        folds_no (int):         the number of folds.
        shuffle (boolean):      shuffle the cycle.

    Yield:
        a sequence of folds.
    """
    folds = list(range(folds_no))
    __folds_iter = shuffled_cycle if shuffle else cycle
    for cnt in strata_cnts:
        for i in islice(__folds_iter(folds), cnt):
            yield i



def stratified_group_folds(strata_cnts,
                           folds_no=10,
                           shuffle=False):
    """Assign elements to folds based on strata counts, in peptide-id groups.

    The strata counts are assumed to be ordered by the user.
    Within each stratum, points are divided into folds in 
    consecutive batches of 'folds_no' points.
    By default, points are prescibed to folds by their order of appearance.

    
    For instance, if retention times were 21.4, 31.5, 53.1, 64.4, 78.2 
    and we wanted 3 folds, then these retention times would be simply mapped to
    folds with numbers 0, 1, 2, 0, 1.
    The 'run_cnts' induce the order of appearance of folds.
    If 'shuffle=True', the numbers will be permuted each time.

    Args:
        strata_cnts (iterable): counts of elements in subsequent strata.
        folds_no (int):         the number of folds.
        shuffle (boolean):      shuffle the cycle.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    elements_cnt = sum(strata_cnts)
    iter_folds   = iter_stratified_folds(strata_cnts,
                                         folds_no,
                                         shuffle)
    return np.fromiter(iter_folds, count=elements_cnt, dtype=np.int8)


def tenzer_folds(peptides_cnt, folds_no=10):
    """Create Tenzer folds without strata.

    There is no additional grouping by peptide-ids here.
    The 'natural randomness' of the data points is used.

    Args:
        peptides_cnt (int): number of peptides
        folds_no (int): the number of folds the data will be divided into.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    return np.fromiter(cycle(range(folds_no)),
                       count=peptides_cnt,
                       dtype=np.int8)


def random_tenzer_folds(peptides_cnt, folds_no=10):
    """Draw folds in Tenszer's windows, but randomly shuffled.
    
    There is no additional grouping by peptide-ids here.

    Args:
        peptides_cnt (int): number of peptides.
        folds_no (int): the number of folds the data will be divided into.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    folds = list(range(folds_no))
    return np.fromiter(shuffled_folds(folds),
                       count=peptides_cnt,
                       dtype=np.int8)


def replacement_folds(peptide_groups_cnt, 
                      folds_no=10):
    """Assign peptide groups to folds independently one from another.

    Args:
        peptide_groups_cnt (int): number of peptide groups.
        folds_no (int): the number of folds the data will be divided into.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    return choice(folds_no,
                  size=peptide_groups_cnt,
                  replace=True)

def replacement_folds_strata(strata_cnts,
                             folds_no=10,
                             shuffle=False):
    """Assign peptide groups to folds independently one from another.

    Args:
        strata_cnts (iterable): numbers of peptide groups in each stratum.
        folds_no (int): the number of folds the data will be divided into.
        shuffle (boolean): for compability only.

    Return:
        out (np.array of ints): the folds prescription for individual peptide groups.
    """
    return replacement_folds(sum(strata_cnts), folds_no)


def set_folds(preprocessed_data,
              feature='rt',
              fold=stratified_group_folds,
              folds_no=10,
              shuffle=True):
    """Assign to folds.

    Args:
        preprocessed_data (pandas.DataFrame): data to assign folds to.
        feature (string):   the name of the feature in the column space of the preprocessed_data that will be aligned.
        fold (function):    the folding function.
        folds_no (int):     the number of folds to split the data into.
        shuffle (boolean):  shuffle the points while folding?
    """
    dp = preprocessed_data
    dp.filter_unfoldable_strata(folds_no)
    if fold.__name__ == 'stratified_group_folds':
        # we want the result to be sorted w.r.t. median rt.
        dp.stats.sort_values(["runs", dp.stat_name + '_' + feature],
                             inplace=True)
    dp.stats['fold'] = fold(dp.strata_cnts, folds_no, shuffle)
    dp.D.drop(labels  = [c for c in dp.D.columns if 'fold' in c], 
              axis    = 1,
              inplace = True)
    dp.D = pd.merge(dp.D, dp.stats[['fold']],
                    left_on='id', right_index=True)
    return dp
