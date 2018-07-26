"""The preprocessing routines."""
import numpy  as np
import pandas as pd


#TODO: replace this with C.
def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)


# TODO: add preprocessing that right now is done in R,
# filtering out 'nonsensical' peptides.
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