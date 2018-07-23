import numpy as np
import pandas as pd


def dedup_sort(x, y, 
               drop_duplicates=True,
               sort=True):
    """Remove dulicate x entries in x and for the corresponding y indices. 
    Sort by x."""
    if drop_duplicates or sort:
        d = pd.DataFrame({'x':x, 'y':y})
        if drop_duplicates:
            d = d.drop_duplicates(subset='x', keep=False)
        if sort:
            d = d.sort_values(['x'])
        return d.x.values, d.y.values
    else:
        return x, y


def get_quantiles(x, chunks_no=4):
    """Calculate chunks_no-quantiles.

    Args:
        x (np.array):   input for which the quantiles are computed.
    Returns:
        np.array: the chunks_no-quantiles.

    """
    return np.percentile(x, np.linspace(0, 100, chunks_no+1))


def percentile_pairs_of_N_integers(N, k):
    """Generate pairs of consecutive k-percentiles of N first integers.

    The distribution concentrates on set { 0 , 1 , .., N-1 }.
    For k = 10, you will get indices of approximate deciles.

    Args:
        N (int): the number of integers.
        k (int): number of approximate percentiles
    Returns:
        iterator: consecutive pairs of k-percentiles.

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
    
    Find approximate percentiles of the uniform distrubution over { 0, 1, .., N - 1}.

    Args:
        N (int):                the number of integers.
        k (int):                number of approximate percentiles
        return_last (logical):  return the last percentile.
        inner (logical):        skip 0 and N-1.
    Returns:
        iterator: consecutive k-percentiles.

    """
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


def overlapped_percentile_pairs(N, k):
    """Provide indices for window-like percentiles.

    Indices defining three consecutive percentile bins.
    First and last are based only on two bins.

    Args:
        N (int):                the number of integers.
        k (int):                number of approximate percentiles

    Returns:
        iterator: consecutive k-percentiles.
    """
    perc = percentiles_of_N_integers(N, k)
    if k > 2:
        i0, i1, i2, i3 = next(perc), next(perc), next(perc), next(perc)
        yield i0, i0, i1, i2
        yield i0, i1, i2, i3
        for i in perc:
            i0, i1, i2, i3 = i1, i2, i3, i
            yield i0, i1, i2, i3
        yield i1, i2, i3, i3
    elif k == 2:
        i0, i1, i2 = next(perc), next(perc), next(perc)
        yield i0, i0, i1, i2
        yield i0, i1, i2, i2
    elif k == 1:
        i0, i1 = next(perc), next(perc)
        yield i0, i0, i1, i1
    else:
        raise ValueError("The number of percentiles must be in {1, 2, .., N}.")

def percentiles_iter(x, k, inner=False):
    """Iterator over approximate percentiles.

    Args:
        x (np.array):       values to calculate the approximate percentiles for.
        k (int):            number of approximate percentiles
        inner (logical):    skip 0 and N-1.
    Returns:
        iterator: consecutive k-percentiles.

    """
    for i in percentiles_of_N_integers(len(x), k, True, inner):
        yield x[i]


def percentiles(x, k, inner=False):
    """Get percentiles in numpy array.

    Args:
        x (np.array):      Array to find percentiles.
        k (int):           Number of approximate percentiles.
        inner (logical):   Skip 0 and N-1.

    """
    assert len(x) > k - 1, "Too many inner percentiles."
    try:
        dtype = x.dtype
    except AttributeError:
        dtype = type(x[0])
    return np.fromiter(percentiles_iter(x, k, inner),
                       dtype=dtype,
                       count=k+1 if not inner else k-1)



