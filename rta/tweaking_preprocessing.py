%load_ext line_profiler
import numpy as np
import pandas as pd

from rta.read_in_data import big_data
from rta.preprocessing import preprocess as preprocess0


min_runs_no = 2
annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
D = annotated

D_id = D.groupby('id')
D_stats = D_id.agg({'rt': np.median, 
                    'mass': np.median,
                    'dt': np.median,
                    'run': ordered_str,
                    'id': len})
D_stats.columns = ['median_rt', 'median_mass', 'median_dt', 'runs', 'runs_no']
D_stats = D_stats[ D_stats.runs_no >= min_runs_no ]
D = pd.merge(D, D_stats, left_on="id", right_index=True)
D = D.assign(rt_median_distance = D.rt - D.median_rt,
			 mass_median_distance = D.mass - D.median_mass,
			 dt_median_distance = D.dt - D.median_dt)



%lprun -f preprocess0 preprocess0(annotated, min_runs_no = 2)
%lprun -f preprocess1 preprocess1(annotated, min_runs_no = 2)

D = annotated
D_id = D.groupby("id")
D_id[['rt', 'dt', 'mass']].transform(np.median)


annotated


D_id.get_group("NVLIFDLGGGTFDVSILTIDDGIFEVK NA")

# FAILED!!!
# I really have enough of pandas.

def test(x = np.array([1, 3, 5, 9]), dims = 10):
	o = np.zeros(dims)
	o[x] = 1
	return np.polyval(o, 2)

%%timeit 
test()


%%timeit
ordered_str(np.array([1, 3, 5, 9]))

numpy.polyval