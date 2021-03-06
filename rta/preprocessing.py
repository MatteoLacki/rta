"""The preprocessing routines."""
import numpy  as np
import pandas as pd

from rta.read.csvs import col_names,\
                          col_names_unlabelled


def split(L, cond):
    """Split the DataFrame accoring to the count in the condition.

    Things with cond > 1 are wrong, == 1 are right.
    """
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
    L = data.dropna(subset=['sequence']).copy()
    L.set_index(['sequence', 'modification', 'run', 'type'], inplace=True)
    L.sort_index(inplace=True)
    # filter peptides that appear multiple time with the same type in the same run
    id_run_type_cnt = L.groupby(L.index).size()
    L, non_unique_id_run_type = split(L, id_run_type_cnt)
    # filter peptides identified in more than one type per run
    L.reset_index(level=['type'], inplace=True)
    L.sort_index(inplace=True)
    types_per_id_run = L.groupby(L.index).size()
    L, non_unique_type_per_id_run = split(L, types_per_id_run)
    # filter out peptides identified with different types in different runs
    L.reset_index(level='run', inplace=True)
    L.sort_index(inplace=True)
    diff_types_diff_runs = L.groupby(L.index).type.nunique()
    L, diff_types_in_diff_runs = split(L, diff_types_diff_runs)
    if return_filtered:
        return L, U, non_unique_id_run_type, non_unique_type_per_id_run, diff_types_in_diff_runs
    else:
        return L, U


def preprocess(annotated_peptides,
               min_runs_no = 5):
    """Preprocess the data.

    Args:
        annotated_peptides (pd.DataFrame): A DataFrame with columns 'id', 'run', and some measuments in other columns. 
        min_runs_no (int): Minimal number of runs a peptide occurs in to qualify for further analysis.

    Returns:
        tuple: preprocessed data, statistics on it, and peptides distibuant
    """
    # work on smaller copy of the data.
    D = annotated_peptides.copy()
    runs = D.run.unique()
    D_id = D.groupby('id')

    ### get basic statistics on D.
    stats = pd.DataFrame({'runs_no': D_id.id.count()})
    stats['runs'] = D_id.run.agg(frozenset)
    assert all(stats.runs.apply(len) == stats['runs_no']), "sets have wrong number of members."

    # peptides distribuant w.r.t. decreasing run appearance
    peptides_in_runs_cnt = stats.groupby('runs_no').size()
    pddra = np.flip(peptides_in_runs_cnt.values, 0).cumsum()
    pddra = {i + 1:pddra[-i-1] for i in range(len(peptides_in_runs_cnt))}

    # filtering out infrequent peptides
    stats = stats[stats.runs_no >= min_runs_no].copy()
    D = D[D.id.isin(stats.index)].copy()
    
    # number of peptides per run
    pepts_per_run = D.groupby('run').id.count()
    pepts_per_run.name = "count"

    return D, stats, pddra, pepts_per_run
