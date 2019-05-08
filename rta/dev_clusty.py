%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import cKDTree as kd_tree
from plotnine import *
from itertools import islice

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.reference import cond_medians
from rta.parse import threshold as parse_thr

from time import time

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")
A['signal2medoid_d'] = np.abs(A[['massa_d', 'rta_d', 'dta_d']]).max(axis=1)
# AW = A[['id','run','rt']].pivot(index='id', columns='run', values='rt')
# 100 * AW.isnull().values.sum() / np.prod(AW.shape) # 80% of slots are free!


class RunChargeGrouper(object):
    """Prepare a sequence of subproblems for the nearest neighbours.

    Args:
        A (pandas.DataFrame): all annotated signals.
        U (pandas.DataFrame): all unannotated signals.
        peptID (str): name of the variable that identifies the peptides in A.
        freevars (list of strings): List of column names that are to be used for finding the nearest neighbours.
    """
    def __init__(self, A, U,
                 peptID = 'id',
                 freevars=['rta', 'dta', 'massa']):
        self.condvars = ['run', 'charge']
        self.freevars = freevars
        self.rq = A[self.condvars].drop_duplicates()
        self.Urq = U.loc[:, freevars + self.condvars].groupby(['run', 'charge'])
        self.Arq = A.loc[:, [peptID] + self.condvars].groupby(['run', 'charge'])
        Aid = A.groupby(peptID)
        self.A_agg = pd.concat([ Aid[freevars].median(), Aid.charge.first()], axis = 1)
        self.A_agg_q = self.A_agg.groupby('charge')

    def __iter__(self):
        """Iterate over medoids of identified peptides and corresponding unindetified signals."""
        for q, r_q in self.rq.groupby('charge'):
            # r_q: a DataFrame with values of runs conditional on charge
            # all identified peptides with charge q: medoids
            A_agg_q = self.A_agg_q.get_group(q)
            for r in r_q['run']:
                Arq = self.Arq.get_group((r,q))
                Urq = self.Urq.get_group((r,q))
                # get all peptide-medoids not in run r with charge q
                A_agg_rq = A_agg_q.loc[~A_agg_q.index.isin(Arq.id), self.freevars]
                if not A_agg_rq.empty:
                    yield A_agg_rq, Urq

def find_nearest_neighbour(grouper, **query_kwds):
    """Find neareast neigbours of medoids of identified peptides within sets of unidentified signals.

    Args:
        grouper (Grouper): a class implementing '__iter__' of pandas.DataFrames of identified and unidentified singals.
        query_kwds: arguments for the cKDtree.query methods.
    Yield:
        DataFrames of unidentified singal attirbuted to particular peptides.
    """
    for A_cond, U_cond in grouper:
        # normalize A_cond U_cond?
        tree = kd_tree(U_cond[grouper.freevars])
        dist, points = tree.query(A_cond, **query_kwds)
        out = U_cond.iloc[points[dist < inf]].reset_index()
        if not out.empty:
            out['d'] = dist[dist < inf]
            out.index = A_cond[dist < inf].index # maybe better to set it as a column?
            yield out

%%time
rq_grouper = RunChargeGrouper(A, U)
NN_rq = find_nearest_neighbour(rq_grouper, k=10, p=inf, distance_upper_bound=1)
NN = pd.concat(NN_rq, axis=0, sort=False)
# 12 secs on old computer : 3.56 secs on a new one.

NN.query('id == "AAAAASLR NA"')
rq_grouper.A_agg.query('id == "AAAAASLR NA"')
A.query('id == "AAAAASLR NA"')

# 2 ways to go now: 
#   optimize the size of the boxes and normalize them to a unit box
#   divide the problem also according to mass groups.
 
# then, we need to check, if some points have not been selected twice for different peptides.

