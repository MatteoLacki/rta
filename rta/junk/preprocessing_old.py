import numpy as np
import pandas as pd
from rta.xvalidation.stratifications_folds import peptide_stratified_folds






def get_stats(D_all, 
              min_runs_no=5,
              var_names=('rt', 'dt', 'mass'), 
              pept_id='id',              
              stat=np.median):
    """Calculate basic statistics conditional on peptide-id.

    Args:
        D_all (pandas.DataFrame): The DataFrame with identified peptides.
        min_runs_no (int): Analyze only peptides that appear at least in that number of runs.
        var_names (iter of strings): names of columns for which to obtain the statistic.
        pept_id (str): name of column that identifies peptides in different runs.
        stat (function): a statistic to apply to the selected features.

    Return:
        D_stats (pandas.DataFrame): A DataFrame summarizing the selected features of peptides. Filtered for peptides appearing at least in a given minimal number of runs. 

    """
    D_id = D_all.groupby('id')
    D_stats = D_id[var_names].agg(stat)
    D_stats.columns = [ stat.__name__ + "_" + vn  for vn in var_names]
    D_stats['runs_no'] = D_id.id.count()
    # this should be rewritten in C.
    D_stats['runs'] = D_id.run.agg(ordered_str)
    return D_stats[D_stats.runs_no >= min_runs_no].copy()



def filter_and_fold(D,
                    D_stats,
                    folds_no=5,
                    fold=peptide_stratified_folds,
                    **kwds):
    """Get folds for cross-validation.

    Trims the supplied data to subsets that can be folded and 
    assigns to folds.
    """
    # sorting by medians facilitates the generation of fold sequences:
        # the tenzer folds are calculated one after another.

    # the 'runs' sort doesn't seem to play any role.
    D_stats.sort_values(["runs", "median_rt"], inplace=True)
    run_cnts = D_stats.groupby("runs").runs.count()
    run_cnts = run_cnts[run_cnts >= folds_no].copy()
    D_stats = D_stats.loc[D_stats.runs.isin(run_cnts.index)].copy()
    # we need sorted DF to append a column correctly
    D_stats['fold'] = fold(run_cnts = run_cnts,
                           folds_no = folds_no)
    # pd.merge copies memory
    D_cv = pd.merge(D,
                    D_stats[['fold']],
                    left_on='id', 
                    right_index=True)
    return D_cv, D_stats, run_cnts


def preprocess(D_all,
               min_runs_no=5,
               folds_no=10,
               var_names = ('rt', 'dt', 'mass'),
               fold=peptide_stratified_folds,
               **filter_and_fold_kwds):
    """Preprocess the data for fitting cross-validated splines."""
    D_stats = get_stats(D_all,
                        min_runs_no)
    D = pd.merge(D_all,
                 D_stats,
                 left_on="id",
                 right_index=True)

    # get median distances
    D.assign(**{n+"_median_distance": D[n] - D['median_'+n] for n in var_names})
    D_cv, D_stats, run_cnts = filter_and_fold(D, 
                                              D_stats,
                                              folds_no,
                                              fold, 
                                            **filter_and_fold_kwds)
    return D_cv, D_stats, run_cnts