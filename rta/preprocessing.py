import numpy as np
import pandas as pd

def preprocess(D, min_runs_no = 5):
    X = D.groupby('id')
    D_stats = pd.DataFrame(dict(runs_no = X.rt.count(),
                                median_rt = X.rt.median(),
                                median_mass = X.pep_mass.median()))
    enough_runs = D_stats.index[ D_stats.runs_no >= min_runs_no ]
    D = D.loc[ D.id.isin(enough_runs) ]
    D = pd.merge(D, D_stats, left_on='id', right_index=True)
    D = D.assign(rt_median_distance = D.rt - D.median_rt)
    return D
