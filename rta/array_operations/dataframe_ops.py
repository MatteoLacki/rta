import numpy as np
import pandas as pd
from itertools import chain


def get_vars(vars, colnames):
	"""Prepare variables for analysis."""
	if not type(vars) == list:
		vars = [vars]
	assert all(v in colnames for v in vars), "Please provide variables that are among the column names."
	return vars


def get_hyperboxes(X, vars, grouping_var='id'):
	"""Get minimal and maximal values of grouped variables.

	Args:
		X (pd.DataFrame): Data with columns for which to calculate the minimal embracing box.
		vars (str or list of strings): columns for which to calcute the medians.
		grouping_var (str): name of the column that defines groups of points.
	"""
	vars = get_vars(vars, X.columns)
	Xg = X.groupby(grouping_var)
	B = pd.concat([Xg[vars].min(), Xg[vars].max()], axis=1)
	B.columns = list(v+s for s in ('_min', '_max') for v in vars)
	B['vol'] = np.ones(B.shape[0])
	for v in vars:
		B[v+'_edge'] = B[v+'_max'] - B[v+'_min']
		B['vol'] *= B[v+'_edge']
	return B


def get_box_log_vol(X, vars, grouping_var='id', return_edges=False):
	"""Get minimal and maximal values of grouped variables.

	Args:
		X (pd.DataFrame): Data with columns for which to calculate the minimal embracing box.
		vars (str or list of strings): columns for which to calcute the medians.
		grouping_var (str): name of the column that defines groups of points.
	"""
	vars = get_vars(vars, X.columns)
	Xg = X.groupby(grouping_var)
	box_edge_len = Xg[variables].max() - Xg[variables].min()
	with np.errstate(divide='ignore'):
		logV = np.log(box_edge_len).sum(axis=1)
	if return_edges:
		return logV, box_edge_len
	else:
		return logV


def conditional_medians(X, vars, grouping_var='id'):
	"""Get the medians conditional on the values in the grouping column.

	Args:
		X (pd.DataFrame): Data with columns for which to calculate the conditional medians.
		vars (str or list of strings): columns for which to calcute the medians.
		grouping_var (str): name of the column that should be conditioned upon.
	"""
	vars = get_vars(vars, X.columns)
	return X.groupby(grouping_var)[vars].median()


def normalize(X, var, value):
    """Add a normalized column to the DataFrame.

    Args:
        X (pd.DataFrame): Contains column 'var'.
        var (str): The name of the column to normalize by 'value'.
        value (float): A non-zero value to normalize by column 'var'.
    """
    assert value != 0, "Pass a non-zero normalization factor."
    X[var + "_n"] = X[var] / value 
