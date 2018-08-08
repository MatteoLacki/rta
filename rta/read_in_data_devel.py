"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

from rta.read_in_data import col_names,\
                             col_names_unlabelled
from rta.isoquant     import retrieve_data
from rta.config       import *

# # the first HELA dataset I've analysed.
# data_path = "../../../Data/annotated_and_unanottated_data.csv"

# # Ute's data-sets for microflow.
# data_path = "~/ms/Matteo/4Ute/2016-141 HYE Microflow_20180716_MF_120min_paper.csv"
# data = pd.read_csv(data_path)


project = "Proj__15272392369260_8293106731954075_100_1"
data    = retrieve_data(password  = password,
                        user      = user,
                        project   = project,
                        verbose   = True)

# data that has not been given any sequence.
U = data.loc[data.sequence.isna(), col_names_unlabelled].copy()

# all the identified peptides.
L = data.dropna(subset = ['sequence']).copy()
# L['id'] = L['sequence'] + " " + L['modification'].astype(str) # old way
L.set_index(['sequence', 'modification', 'run', 'type'], inplace = True)

# filter out peptides that have more than one instance while conditioning on id, run and type.
# eliminate multiple peptides in the same run with the same id.
L.sort_index(inplace = True)
id_run_type_cnt = L.groupby(L.index).size()
non_unique_id_run_type = L[ L.index.isin(id_run_type_cnt[id_run_type_cnt > 1].index) ]
L = L[ L.index.isin(id_run_type_cnt[id_run_type_cnt == 1].index) ]

# filter peptides identified in more than one type per run
L.reset_index(level = ['type'], inplace = True)
types_per_id_run = L.groupby(L.index).size()
non_unique_type_per_id_run = L[ L.index.isin(types_per_id_run[types_per_id_run > 1].index) ]
L = L[ L.index.isin(types_per_id_run[types_per_id_run == 1].index) ]

# filter out peptides identified with different types in different runs
L.reset_index(level = 'run', inplace = True)

diff_types_diff_runs = L.groupby(L.index).type.nunique()
peps_diff_types_in_diff_runs = L[ L.index.isin(diff_types_diff_runs[diff_types_diff_runs > 1].index)]
L = L[ L.index.isin(diff_types_diff_runs[diff_types_diff_runs == 1].index)]

