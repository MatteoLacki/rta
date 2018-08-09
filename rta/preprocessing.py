"""The preprocessing routines."""
import numpy  as np
import pandas as pd

from rta.read_in_data import col_names,\
                             col_names_unlabelled

#TODO: replace this with C.
def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)


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


# It does make sense to introduce convention, that the annotated peptides DataFrame always has 
# columns: run, charge, sequence, and modification.
def preprocess(annotated_peptides,
               min_runs_no = 5,
               var_names   = ('rt', 'dt', 'mass'),
               pept_id     = 'id',
               run_name    = 'run',
               charge_name = 'charge'):
    """Preprocess the data.

    Args:
        annotated_peptides (pd.DataFrame): A DataFrame with identified peptides.
        min_runs_no (int):                 Minimal number of runs a peptide occurs in to qualify for further analysis.
        var_names (iter of strings):       Names of columns for which to obtain the statistic.
        pept_id (str):                     Name of the column that identifies peptides in different runs.
        run_name (str):                    Name of the column with runs.
        charge_names (str):                Name of the column with charges.

    Returns:
        Peptides: a named tuple with fields 'data', 'statistics', 'runs' and 'distribuant'.
    """
    all_col_names = list((pept_id, run_name, charge_name) + var_names)
    assert all(vn in annotated_peptides.columns for vn in all_col_names),\
             "Not all variable names are among the column names of the supplied data frame."

    # work on smaller copy of the data.
    D = annotated_peptides[all_col_names].copy()
    runs = D[run_name].unique()
    D_id = D.groupby('id')

    ### get basic statistics on D.
    stats = pd.DataFrame({'runs_no': D_id.id.count()})
    # this should be rewritten in C.
    stats['runs'] = D_id.run.agg(ordered_str)
    # counting unique charge states per peptide group
    stats['charges'] = D_id.charge.nunique()
    peptides_in_runs_cnt = stats.groupby('runs_no').size()

    # peptides distribuant w.r.t. decreasing run appearance
    pddra = np.flip(peptides_in_runs_cnt.values, 0).cumsum()
    pddra = {i + 1:pddra[-i-1] for i in range(len(peptides_in_runs_cnt))}

    # filtering out infrequent peptides
    stats = stats[stats.runs_no >= min_runs_no].copy()
    D = D[ D[pept_id].isin(stats.index) ].copy()

    return dict(data             = D,
                stats            = stats,
                runs             = runs,
                quasidistribuant = pddra,
                var_names        = var_names,
                pept_id          = pept_id,
                run_name         = run_name,
                charge_name      = charge_name)


def filter_unfoldable(peptides, folds_no = 10):
    """Filter peptides that cannot be folded.

    Trim both stats on peptides and run data of all peptides
    that are not appearing in the same runs in a group of at
    least 'folds_no' other peptides.

    Args:
        folds_no (int): the number of folds to divide the data into.
    """
    s = peptides['stats']
    # count how many peptides appear together in the same runs, e.g. run 1, 2, 4.
    # Peptides appearing together in a given set of runs form a stratum.
    strata_cnts = s.groupby("runs").runs.count()
    # Filter strata that have less than folds_no peptides.
    # E.G. if only 4 peptides appear in runs 1, 2, and 4, then we cannot 
    # meaningfully place them in 10 different folds. 
    # We could place them in at most 4 folds.
    strata_cnts             = strata_cnts[strata_cnts >= folds_no].copy()
    peptides['strata_cnts'] = strata_cnts
    # filter out peptides groups that cannot take part in the folding.
    s = s[ np.isin(s.runs, strata_cnts.index) ].copy()
    peptides['stats'] = s
    # filter out individual peptides (in different runs) that cannot take part in the folding.
    data = peptides['data']
    data = data[ data[ peptides['pept_id'] ].isin(s.index) ].copy()
    peptides['data'] = data
    return peptides