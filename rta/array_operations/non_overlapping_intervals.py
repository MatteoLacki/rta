import numpy as np


class Interval(object):
    def __init__(self, L, R):
        assert all(L < R)
        self.LR = np.zeros(dtype=float, shape=(L.shape[0]+R.shape[0],))
        self.LR[0::2] = L
        self.LR[1::2] = R
        self.L = L
        self.R = R

    def __getitem__(self, x):
        raise NotImplementedError


class OpenClosed(Interval):
    def __getitem__(self, x, idx=True):
        i = np.searchsorted(self.LR, x, side='left')
        cluster_no, in_clust = np.divmod(i, 2)
        cluster_no = np.where(in_clust, cluster_no, -1)
        return cluster_no


class ClosedOpen(Interval):
    def __getitem__(self, x):
        i = np.searchsorted(self.LR, x, side='right')
        cluster_no, in_clust = np.divmod(i, 2)
        cluster_no = np.where(in_clust, cluster_no, -1)
        return cluster_no


class OpenOpen(Interval):
    def __getitem__(self, x):
        i = np.searchsorted(self.LR, x, side='left')
        cluster_no, in_clust = np.divmod(i, 2)
        not_right_end = x < np.take(self.R, cluster_no, mode='clip')
        cluster_no = np.where(np.logical_and(in_clust, not_right_end), cluster_no, -1)
        return cluster_no


def test_non_overlapping_intervals():
    L = np.array([0, 10])
    R = np.array([1, 11])
    OC = OpenClosed(L, R)
    x = (-1, 0, .5, 1, 2, 10, 10.5, 11, 12)
    assert all(OC[x] == [-1,-1, 0, 0,-1,-1, 1, 1,-1])
    CO = ClosedOpen(L, R)
    assert all(CO[x] == [-1, 0, 0,-1,-1, 1, 1,-1,-1])
    OO = OpenOpen(L, R)
    assert all(OO[x] == [-1,-1, 0,-1,-1,-1, 1,-1,-1])
