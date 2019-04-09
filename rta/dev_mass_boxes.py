from bisect import bisect
import numpy as np


class LightweightSpectrum(object):
    def __init__(self, min_mz, max_mz, total_intensities):
        self.clust_no = len(min_mz)
        self._spec = np.zeros(dtype = float,
                              shape = (self.clust_no*2,))
        # real spectrum
        self._spec[0::2] = min_mz
        self._spec[1::2] = max_mz
        self.intensity = total_intensities

    def __getitem__(self, key):
        i = bisect(self._spec, key)
        i_div, i_mod = divmod(i, 2)
        return i_div if i_mod else -1

    def __repr__(self):
        return self._spec.__repr__()

    def min_max_mz(self, i):
        return tuple(self._spec[(2*i):(2*i+2)])



class SpectralIntervals(LightweightSpectrum):
    def __init__(self, left_mz, right_mz):
        """Quicker and vectorized version of LightweightSpectrum.
    
        Args:
            left_mz (np.array): left ends of the intervals
            right_mz (np.array): right ends of the intervals
        """
        self.clust_no = len(left_mz)
        self._spec = np.zeros(dtype = float,
                              shape = (self.clust_no*2,))
        # real spectrum
        self._spec[0::2] = left_mz
        self._spec[1::2] = right_mz

    def __getitem__(self, mz):
        w = self._spec
        i = np.searchsorted(w, mz, side='left')
        # j = np.searchsorted(w, mz, side='left')
        # return np.where(i == j, np.floor_divide(i, 2), -1)
        i_div, i_mod = np.divmod(i, 2)
        return np.where(i_mod, i_div, -1)



class OpenIntervals(LightweightSpectrum):
    def __init__(self, left_mz, right_mz):
        """Quicker and vectorized version of LightweightSpectrum.
    
        Args:
            left_mz (np.array): left ends of the intervals
            right_mz (np.array): right ends of the intervals
        """
        self.clust_no = len(left_mz)
        self._spec = np.zeros(dtype = float,
                              shape = (self.clust_no*2,))
        # real spectrum
        self._spec[0::2] = left_mz
        self._spec[1::2] = right_mz

    def __getitem__(self, mz):
        il = np.searchsorted(self._spec, mz, side='left')
        il_div_2, il_mod_2 = np.divmod(il, 2)
        out = np.where(
            il_mod,
            np.where(
                np.where(
                    il < len(w),
                    mz < np.take(self._spec, il, mode='clip'),
                    False
                ),
                il_mod_2,
                -1
            ),
            -1
        )
        return out

oi = OpenIntervals([0, 6], [3, 10])
oi._spec
oi[mz]

# check larger then last element.
mz = np.array([1, 2, 9, 13.5, 13.6, 3, -10])
intervals = [(1,5), (8,10), (13,14)]

def in_closed_intervals(mz, intervals):
    left_mz, right_mz = (np.array(l) for l in zip(*intervals))
    i = np.searchsorted(right_mz, mz, side='left')
    out = np.full(mz.shape, -1)
    smaller = mz <= right_mz[-1]
    out[smaller] = np.where(np.take(left_mz, i[smaller]) <= mz[smaller],
                            i[smaller], -1)
    return out

%%timeit
mz = np.array([0, 1, 2, 9, 13.5, 13.6, 3, 50])
intervals = [(1,5), (8,10), (13,14)]
x = in_closed_intervals(mz, intervals)








left_mz[i]

il_div_2, il_mod_2 = np.divmod(il, 2)
ir_div_2, ir_mod_2 = np.divmod(ir, 2)

# (l, r)
np.where(
    il_mod,
    np.where(
        np.where(
            il < len(w),
            mz < np.take(w, il, mode='clip'),
            False
        ),
        il_mod_2,
        -1
    ),
    -1
)




# [l, r]
np.where(
    il_mod,
    np.where(
        np.where(
            il < len(w),
            mz < np.take(w, il, mode='clip'),
            False
        ),
        il_mod_2,
        -1
    ),
    -1
)






np.take([0,2,5], [0,0,0,1])



i_div, i_mod
np.where(i_mod, i_div, -1)
np.isin(mz, w)

mz = np.array([-1, 0, .5, 1, 1.2])
L = SpectralIntervals([0,5, 8], 
                      [1,6,10])
w = L._spec
mz = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
L[mz]






l = np.array([0, 5])
r = np.array([2, 6])

np.array(list(zip(l,r)))


np.searchsorted([-inf, 1,2,3,4,5], 1, side='left')


np.searchsorted([-inf, 1,2,3,4,5], 1, side='left')

mz = mz[mz >= 0]
def classify(mz, w):
    i = np.searchsorted(w, mz, side='left')
    i_div, i_mod = np.divmod(i, 2)
    return np.where(i_mod, i_div, -1)

np.where(mz < w[0], -1, classify(mz, w))


np.searchsorted(w, mz, side='left')


mz = np.array([0, .5, 1, 1.5, 4, 5, 5.5, 6, 10])


def lightweight_spectrum(min_mz, max_mz, total_intensities):
    return LightweightSpectrum(min_mz,
                               max_mz,
                               total_intensities)


def test_lightweight_spectrum():
    L = lightweight_spectrum([0,5,8], 
                             [1,6,9],
                             [3,4,5])
    assert L[-2] == -1
    assert L[0]  ==  0
    assert L[.5] ==  0
    assert L[1]  == -1
    assert L[1.5]== -1
    assert L[4]  == -1
    assert L[5]  ==  1
    assert L[5.5]==  1
    assert L[6]  == -1
    assert L[10] == -1
