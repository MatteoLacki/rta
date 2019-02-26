import numpy as np


def xyg_iter(x, y, g):
	for gr in np.unique(g):
		yield x[g == gr], y[g == gr], gr
