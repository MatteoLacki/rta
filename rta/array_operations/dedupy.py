"""Here you will find functions for getting rid of the same data points."""
from math import inf
import numpy as np


def dedup(x, y):
	"""Deduplicate the arguments for fitting.
	
	Args:
		x (iterable): values to dedup.
		y (iterable): corresponding values.
	Returns:
		tuple with values from the input, but retaining only the first
		among the repeating ones.
	"""
	x1 = []
	y1 = []
	Xprev = -inf
	for X,Y in zip(x,y):
		if X > Xprev:
			x1.append(X)
			y1.append(Y)
		Xprev = X
	return x1, y1


def dedup_np(x,y):
	"""Deduplicate the arguments for fitting.
	
	Much faster then dedup.

	Args:
		x (np.array): values to dedup.
		y (np.array): corresponding values.
	Returns:
		tuple with values from the input, but retaining only the first
		among the repeating ones.
	"""
	o = np.full(x.shape, True)
	o[1:] = np.diff(x) > 0
	return x[o], y[o]


