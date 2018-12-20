import numpy.random as npr

def runif(n, a=0, b=1):
	"""Draw n independent observations from U(a,b).

	Args:
		a (float): minimal value.
		b (float): maximal value.
	"""
	return npr.random(n)*(b-a)+a


