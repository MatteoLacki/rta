import numpy as np


def get_quantiles(x, chunks_no=4):
    """A silly function."""
    return np.percentile(x, np.linspace(0, 100, chunks_no+1))


def percentile_pairs_of_N_integers(N, k):
    """Generate pairs of consecutive k-percentiles of N first integers.

    The distribution concentrates on set { 0 , 1 , .., N-1 }.
    For k = 10, you will get indices of approximate deciles.
    """
    assert N >= 0 and k > 0
    base_step = N // k
    res = N % k
    s = 0 # start
    e = 0 # end
    for _ in range(1, k+1):
        e += base_step
        if res > 0:
            e += 1
            res -= 1
        yield s, e
        s = e


def percentiles_of_N_integers(N, k, return_last=True, inner=False):
    """Generate k-percentiles of uniform distribution over N elements.
    
    Find approximate percentiles of the uniform distrubution over { 0, 1, .., N - 1}."""
    assert N >= 0 and k > 0
    base_step = N // k
    res = N % k
    o = 0
    if not inner:
        yield o
    for _ in range(k-1):
        o += base_step
        if res > 0:
            o += 1
            res -= 1
        yield o
    if not inner and return_last:
        yield N-1


def percentiles_iter(x, k, inner=False):
    for i in percentiles_of_N_integers(len(x), k, True, inner):
        yield x[i]


def percentiles(x, k, inner=False):
    try:
        dtype = x.dtype
    except AttributeError:
        dtype = type(x[0])
    return np.fromiter(percentiles_iter(x, k, inner),
                       dtype=dtype,
                       count=k+1 if not inner else k-1)



