import bisect
from collections import namedtuple, Counter, defaultdict
import numpy as np
import pandas as pd
import intervaltree as it
import networkx as nx


D = pd.read_csv('rta/data/denoised_data.csv')
D_signal = D[D.signal == 'signal']
peptide_grouping = D_signal.groupby('id')
signal_runs_no = peptide_grouping.rt.count()

min_runs_no = 2
D_signal_min_cnts = D_signal[D_signal.id.isin(signal_runs_no[signal_runs_no >= min_runs_no].index)]
eps = np.finfo(float).eps




class Multivariate_Tree(object):
    def __init__(self, data, id_column_name, column_names,
                 stats = (np.mean, np.median),
                 eps = 10e-6):
        self.names = column_names
        self.tree = {var: it.IntervalTree() for var in self.names}
        self.peps = {c: {} for c in self.names}
        for id, d in data.groupby(id_column_name):
            for var, tree in self.tree.items():
                min_val = min(d[var])
                max_val = max(d[var])
                if min_val == max_val:
                    min_val -= eps
                    max_val += eps
                self.tree[var][min_val:max_val] = id
                self.peps[var][id] = (min_val, max_val)

    def get(self, **x):
        return set.intersection(*(set(e.data for e in self.tree[k][v])
                                  for k, v in x.items() ))

    def __getitem__(self, x):
        return {n : self.peps[n][x] for n in self.names}
