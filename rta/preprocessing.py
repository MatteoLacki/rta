import numpy as np
import pandas as pd


# TODO introduce changes that will cope with the charges:
# this cannot be a median based approach.
# 
# TODO eliminate the 'id' column and use indexing based on 
# sequence and modification instead (to save memory)

def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)

def preprocess(D, min_runs_no = 5):
    D_id = D.groupby('id')
    D_stats = D_id.agg({'rt': np.median, 
                        'mass': np.median,
                        'dt': np.median,
                        'run': ordered_str,
                        'id': len})
    D_stats.columns = ['median_rt', 
                       'median_mass',
                       'median_dt',
                       'runs',
                       'runs_no']
    D_stats = D_stats[ D_stats.runs_no >= min_runs_no ]
    
    D = pd.merge(D, D_stats, left_on="id", right_index=True)
    D = D.assign(rt_median_distance = D.rt - D.median_rt,
                 mass_median_distance = D.mass - D.median_mass,
                 dt_median_distance = D.dt - D.median_dt)

    return D, D_stats
