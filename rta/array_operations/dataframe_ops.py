import pandas as pd


def get_hyperboxes(X, vars, grouping_var='id'):
	"""Get minimal and maximal values of grouped variables."""
	if not type(vars) == list:
		vars = [vars]
	cols = []
	names = []
	vid = X.groupby(grouping_var)
	for var in vars:
		v = vid[var]
		cols.append(v.min())
		cols.append(v.max())
		names.append(var+"_min")
		names.append(var+"_max")
	B = pd.DataFrame(pd.concat(cols, axis=1))
	B.columns = names
	return B
