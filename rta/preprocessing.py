import numpy as np
import pandas as pd
from rta.xvalidation.stratifications_folds import peptide_stratified_folds



# TODO: have to generalize the selection of column to DT from only RT
def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)


# Still overally slow.
def get_stats(D_all, min_runs_no = 5):
    D_id = D_all.groupby('id')
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
    return D_stats[D_stats.runs_no >= min_runs_no].copy()


def get_medians(D_all, D_stats, min_runs_no = 5):
    """Calculate distance to medians for 'rt', 'mass', 'dt'."""
    D = pd.merge(D_all, D_stats, left_on="id", right_index=True)
    D = D.assign(rt_median_distance   = D.rt - D.median_rt,
                 mass_median_distance = D.mass - D.median_mass,
                 dt_median_distance   = D.dt - D.median_dt)
    return D


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
               fold=peptide_stratified_folds,
               **filter_and_fold_kwds):
    """Preprocess the data for fitting cross-validated splines."""
    D_stats = get_stats(D_all, min_runs_no)
    D = get_medians(D_all, D_stats, min_runs_no)
    D_cv, D_stats, run_cnts = filter_and_fold(D, D_stats, folds_no, fold, 
                                              **filter_and_fold_kwds)
    return D_cv, D_stats, run_cnts