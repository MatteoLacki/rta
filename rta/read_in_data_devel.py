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
                        ip        = ip,
                        project   = project,
                        verbose   = True)


def split(L, cond):
    wrong = L[ L.index.isin(cond[cond  > 1].index) ]
    good  = L[ L.index.isin(cond[cond == 1].index) ]
    return good, wrong

def filter_peptides_with_unique_types(data, return_filtered = True):
    """Filter out peptides that appear in more than one type per run.

    Also, find peptides that appear with in different types across different runs.

    Args:
        data (pd.DataFrame):       Mass Project data.
        return_filtered (logical): Return the filtered out peptides too.
    Returns:
        tuple : filtered peptides and unlabelled petides. Optionally, filtered out peptides, too.
    """
    # data that has not been given any sequence.
    U = data.loc[data.sequence.isna(), col_names_unlabelled].copy()
    # all the identified peptides.
    L = data.dropna(subset = ['sequence']).copy()
    L.set_index(['sequence', 'modification', 'run', 'type'], inplace = True)
    L.sort_index(inplace = True)
    # filter peptides that appear multpile time with the same type in the same run
    id_run_type_cnt = L.groupby(L.index).size()
    L, non_unique_id_run_type = split(L, id_run_type_cnt)
    # filter peptides identified in more than one type per run
    L.reset_index(level = ['type'], inplace = True)
    L.sort_index(inplace = True)
    types_per_id_run = L.groupby(L.index).size()
    L, non_unique_type_per_id_run = split(L, types_per_id_run)
    # filter out peptides identified with different types in different runs
    L.reset_index(level = 'run', inplace = True)
    L.sort_index(inplace = True)
    diff_types_diff_runs = L.groupby(L.index).type.nunique()
    L, diff_types_in_diff_runs = split(L, diff_types_diff_runs)
    if return_filtered:
        return L, U, non_unique_id_run_type, non_unique_type_per_id_run, diff_types_in_diff_runs
    else: # HAHAHA, LU decomposition - FU!!!
        return L, U

L, U = filter_peptides_with_unique_types(data, False)

# L['id'] = L['sequence'] + " " + L['modification'].astype(str) # old way