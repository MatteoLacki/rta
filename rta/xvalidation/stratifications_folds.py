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


def peptide_stratified_folds(peptides_cnt, run_cnts, folds_no):
    folds = np.zeros(peptides_cnt, dtype=np.int8)
    s = e = 0
    for cnt in run_cnts:
        e += cnt
        folds[s:e] = K_folds(cnt, folds_no)
        s = e
    return folds