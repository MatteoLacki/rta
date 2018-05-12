from collections import Counter
import numpy as np
import pandas as pd
from sklearn import cluster
import networkx as nx

from rta.misc import max_space
from rta.preprocessing import preprocess
from rta.read_in_data import DT as D
from rta.splines.denoising import denoise_and_align
from rta.kd_tree.kd_tree_nice import build_kdtree

formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
DF = preprocess(D, min_runs_no = 2)
# Removing the peptides that have their RT equal to the median.
# TODO: think if these points cannot be directly modelled.
# and what if we start moving them around?
# WARNING: skipping this for now.
# DF = DF[DF.rt_median_distance != 0]

# All points are 'signal' before denoising: guardians
DF['signal'] = 'signal'
# Division by run number
runs = list(data for _, data in DF.groupby('run'))
models = list(denoise_and_align(runs, formula))

# resemble all the data sets
DF_2 = pd.concat(d for m, d in models)
DF_2.head()

# The noise seems to be confirmed by the analysis of the drift times.
[{g: np.median(np.abs(x.dt_median_distance))
  for g, x in d.groupby('signal')}
 for _, d in models]

[{g: np.median(np.abs(x.le_mass_median_distance))
  for g, x in d.groupby('signal')}
 for _, d in models]

# Trying out the modelling of the percentiles of the data.

# Calculate the spaces in different directions for signal points
DF_2_signal = DF_2[DF_2.signal == 'signal']
X = DF_2_signal.groupby('id')
D_stats = pd.DataFrame(dict(runs_no_aligned         = X.rt.count(),
                            rt_aligned_median       = X.rt.median(),
                            rt_aligned_min          = X.rt.min(),
                            rt_aligned_max          = X.rt.max(),
                            rt_aligned_max_space    = X.rt_aligned.aggregate(max_space),
                            pep_mass_max__space     = X.pep_mass.aggregate(max_space),
                            le_mass_max_space       = X.le_mass.aggregate(max_space),
                            le_mass_aligned_min     = X.le_mass.min(),
                            le_mass_aligned_max     = X.le_mass.max(),
                            dt_max_space            = X.dt.aggregate(max_space),
                            dt_aligned_min          = X.dt.min(),
                            dt_aligned_max          = X.dt.max()))
points = [((r.rt_aligned, r['dt'], r.le_mass), r.id) for _, r in DF_2_signal.iterrows()]
tree = kdtree(points)
# # Testing the search.
# box = np.array([[30,    40],
#                 [25,    27],
#                 [600, 1000]])
# proteins_in_box = tree.box_search(box)


# Making the neighbourhood graph.
G = nx.Graph()
for idx1, r in D_stats.iterrows():
    box = np.array([[r.rt_aligned_min,      r.rt_aligned_max],
                    [r.le_mass_aligned_min, r.le_mass_aligned_max],
                    [r.dt_aligned_min,      r.dt_aligned_max]])
    G.add_node(idx1)
    for idx2 in tree.box_search(box):
        G.add_node(idx2)
        G.add_edge(idx1, idx2)

# %^&!#@$@# SOO FUCKING FAST!!!!
cc_len = [len(cc) for cc in nx.connected_components(G)]
Counter(cc_len)
