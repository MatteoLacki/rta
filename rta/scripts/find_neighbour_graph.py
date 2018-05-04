from collections import namedtuple, Counter, defaultdict
import numpy as np
import pandas as pd
import intervaltree as it
import networkx as nx
from sklearn.datasets import make_blobs
import pickle

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


X, y = make_blobs(n_samples = 3000,
                  n_features = 3,
                  centers = 30)

X = pd.DataFrame(X)
X.columns = ['rt_aligned', 'dt', 'le_mass']
X['id'] = y

MT = Multivariate_Tree(X, 'id', ('rt_aligned', 'dt', 'le_mass'))

peptides = set(X.id)
G = nx.Graph()

while len(peptides) > 0:
    pep = peptides.pop()
    G.add_node(pep)
    for other_pep in MT.get(**{k: it.Interval(v[0], v[1])
                               for k, v in MT[pep].items()}):
        G.add_node(other_pep)
        G.add_edge(pep, other_pep)


with open('rta/data/graph.pickle', 'wb') as h:
    pickle.dump(G, h)
