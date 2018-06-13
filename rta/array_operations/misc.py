def get_quantiles(x, chunks_no=4):
	"""A silly function."""
	return np.percentile(x, np.linspace(0, 100, chunks_no+1))


def get_k_percentile_pairs_of_N_integers(N, k):
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
