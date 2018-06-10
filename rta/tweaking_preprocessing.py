%load_ext line_profiler
import numpy as np
import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess as preprocess0


annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")

def preprocess1(D,
                min_runs_no  = 5):
	D_id = D.groupby(id)
	D_stats = D_id.agg({'rt': np.median, 
					    'mass': np.median,
					    'dt': np.median})
	D_stats.rename({'rt': 'median_rt',
					'mass': 'median_mass',
					'dt': 'median_dt'})
	D_stats = D_stats.assign(runs_no=D_id.id.count())
	enough_runs = D_stats.index[ D_stats.runs_no >= min_runs_no ]
	print(D.rt)
    D = D.loc[ D.id.isin(enough_runs) ]
    D = pd.merge(D, D_stats, left_on='id', right_index=True)
    D = D.assign(rt_median_distance     = D.rt   - D.median_rt)
    D = D.assign(mass_median_distance   = D.mass - D.median_mass)
    D = D.assign(dt_median_distance     = D.dt   - D.median_dt)
    return D, D_stats

%lprun -f preprocess0 preprocess0(annotated, min_runs_no = 2)
%lprun -f preprocess1 preprocess1(annotated, min_runs_no = 2)

D = annotated
D_id = D.groupby("id")
D_id[['rt', 'dt', 'mass']].transform(np.median)


D_id.get_group("NVLIFDLGGGTFDVSILTIDDGIFEVK NA")

# FAILED!!!
# I really have enough of pandas.
