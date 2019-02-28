from rta.reference import stat_reference


def Tenzerize(X, n, a):
    """Perform a hunt for correct alignment.

	Args:
		X (pd.DataFrame): DataFrame with columns x, y (reference), and run.
		n (int): number of repeated fittings.
		a (Aligner): an initialized aligner.
    """
    for i in range(n):
        a.fit(X)
        x = a(X)
        X.rename(columns={'x':'x'+str(i), 'y':'y'+str(i)}, inplace=True)
        X['x'] = x
        X = stat_reference(X)
    X.rename(columns={'x':'x'+str(n), 'y':'y'+str(n)}, inplace=True)
    return a, X


def Matteotti(X, a):
	"""A simple strategy.

	Args:
		X (pd.DataFrame): DataFrame with columns x, y (reference), and run.
		a (Aligner): an initialized aligner.
	"""
	a.fit(X)
	X['x_aligned'] = a.fitted()
	return a, X



def IterativeReferenceHunt():
	"""Perform iterative reference hunt."""
	pass
