from math import sqrt, ceil, floor


def max_key_val(d):
	"""Return the key-value with the maximal value.
	Args:
		d (dict-like): A dictionary.
	"""
	return max(d.items(), key=lambda x: x[1])


def plot_matrix_sizes(plots_no):
	"""Calculate the size of the matrix fitting 'plots_no' plots.

	Args:
		plots_no (int): the number of plots in the matrix.
	Returns:
		tuple containing the number of row 'rows_no' and columns 'cols_no'.
	"""
	rows_no = floor(sqrt(plots_no))
	cols_no = rows_no + ceil((plots_no - rows_no**2)/rows_no)
	return rows_no, cols_no
