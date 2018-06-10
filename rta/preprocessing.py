import numpy as np
import pandas as pd


# TODO introduce changes that will cope with the charges:
# this cannot be a median based approach.
# 
# TODO eliminate the 'id' column and use indexing based on 
# sequence and modification instead (to save memory)

def ordered_str(ints):
    x = list(ints)
    x.sort()
    return "_".join(str(i) for i in x)


def preprocess(D,
               min_runs_no  = 5,
               rt           = 'rt',
               mass         = 'mass',
               dt           = 'dt'):
    X = D.groupby('id')
    D_stats = pd.DataFrame(dict(runs_no     = X[rt].count(),
                                median_rt   = X[rt].median(),
                                median_mass = X[mass].median(),
                                median_dt   = X[dt].median()))
    enough_runs = D_stats.index[ D_stats.runs_no >= min_runs_no ]
    D = D.loc[ D.id.isin(enough_runs) ]
    D = pd.merge(D, D_stats, left_on='id', right_index=True)
    D = D.assign(rt_median_distance     = D[rt]   - D.median_rt)
    D = D.assign(mass_median_distance   = D[mass] - D.median_mass)
    D = D.assign(dt_median_distance     = D[dt]   - D.median_dt)
    return D, D_stats
