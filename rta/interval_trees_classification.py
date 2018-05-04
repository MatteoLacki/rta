import bisect
from collections import namedtuple, Counter, defaultdict
import numpy as np
import pandas as pd
import intervaltree as it



D = pd.read_csv('rta/data/denoised_data.csv')
D_signal = D[D.signal == 'signal']
peptide_grouping = D_signal.groupby('id')
signal_runs_no = peptide_grouping.rt.count()

min_runs_no = 2
D_signal_min_cnts = D_signal[D_signal.id.isin(signal_runs_no[signal_runs_no >= min_runs_no].index)]
eps = np.finfo(float).eps


x = tuple([{}] * 3)
x = tuple({} for _ in range(3))

x[0]['a'] = 120
x


# class Multivariate_Tree(object):
#     def __init__(self, data, id_column_name, column_names,
#                  stats = (np.mean, np.median),
#                  eps = 10e-6):
#         STATS = namedtuple('stats', [s.__name__ for s in stats])
#         self.tree = {var: it.IntervalTree() for var in column_names}
#         {c: {} for c in column_names}
#         for id, d in data.groupby(id_column_name):
#             for var, tree in self.tree.items():
#                 min_val = min(d[var])
#                 max_val = max(d[var])
#                 if min_val == max_val:
#                     min_val -= eps
#                     max_val += eps
#                 self.tree[var][min_val:max_val] = id
#                 self.pep_stats[var][id] = STATS(*(s(d[var]) for s in stats))
#
#     def __getitem__(self, **x):
#         return set.intersection( self.tree[k][v] for k, v in x )

class Multivariate_Tree(object):
    def __init__(self, data, id_column_name, column_names,
                 stats = (np.mean, np.median),
                 eps = 10e-6):
        self.tree = {var: it.IntervalTree() for var in column_names}
        for id, d in data.groupby(id_column_name):
            for var, tree in self.tree.items():
                min_val = min(d[var])
                max_val = max(d[var])
                if min_val == max_val:
                    min_val -= eps
                    max_val += eps
                self.tree[var][min_val:max_val] = id

    def get(self, **x):
        return set.intersection([self.tree[k][v] for k, v in x ])

#
#
# class Multivariate_Tree(object):
#     def __init__(self, data, id_column_name, column_names,
#                  stats = (np.mean, np.median),
#                  eps = 10e-6):
#
#         self.trees = [it.IntervalTree() for _ in column_names]
#         for id, d in data.groupby(id_column_name):
#             for c, tree in zip(column_names, self.trees):
#                 min_val = min(d[c])
#                 max_val = max(d[c])
#                 if min_val == max_val:
#                     min_val -= eps
#                     max_val += eps
#                 tree[min_val:max_val] = id
#
#     def __getitem__(self, *x):
#         querries = [set(e.data for e in tree[v]) for v, tree in zip(x, self.trees)]
#         return set.intersection(*querries)


MT = Multivariate_Tree(D_signal_min_cnts, 'id', ('rt_aligned', 'dt', 'le_mass'))
MT.get(rt_aligned = 100)
