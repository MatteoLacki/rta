def max_key_val(d):
	"""Return the key-value with the maximal value.
	Args:
		d (dict-like): A dictionary.
	"""
	return max(d.items(), key=lambda x: x[1])

