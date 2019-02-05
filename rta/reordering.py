%load_ext autoreload
%autoreload 2

import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)

annotated_all, unlabelled_all = big_data()
D, stats, pddra = preprocess(annotated_all, 5)

from collections import Counter
x = Counter(stats.runs)


# folding: not part of calibrator itself.
D.groupby
stats

D.set_index('id', inplace=True)


D.groupby(lambda x: 'it is 10' if x==10 else 'not 10').count()
D[['id', 'run']]



D.groupby(x).count()

# I need a function applicable to a DataFrame
list(stats.groupby("runs"))
list(K_folds(16, 5))

# work on stats? sounds nice: at least we get rid of the peptide groups level, which should be good for the analysis

from rta.cv.folds import K_folds


def drop(X, col):
	try:
		X.drop(columns=col, inplace=True)
	except KeyError:
		pass


def trivial_K_folds(D, K):
	"""This simply adds folds independent of protein-groups or any strata.

	Args:
		D (pd.DataFrame): The data with measurements with colund id.
		K (int): The number of folds.
	"""
	drop(D, 'fold')
	D['fold'] = K_folds(len(D), K)
	return D


def stratified_grouped_fold(D, stats, K, strata='runs'):
	"""Add a folding to the data.

	Args:
		D (pd.DataFrame): The data with measurements with colund id.
		stats (pd.DataFrame): Statistics of data D, containing per peptide info.
		K (int): The number of folds.
		strata (str): String with the name of the column defining the strata in stats.
	"""
	drop(D, 'fold')
	drop(stats, 'fold')

	# thie gets called on every group
	def f(g):
		g['fold'] = K_folds(len(g), K)
		return g

	# gettin group-stratum specific fold assignments
	stats = stats.groupby(strata).apply(f)

	# passing it to main data
	D = D.join(stats.fold, on='id')

	return D, stats

D, stats = stratified_grouped_fold(D, stats, 3)
D, stats = stratified_grouped_fold(D, stats, 3, "runs_no")











# what if group has less elements then needed?
K_folds(3, 5)
# simply we attribute points to one of the things they can go.