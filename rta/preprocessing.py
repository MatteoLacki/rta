import numpy as np
import pandas as pd
from rta.xvalidation.cross_validation import peptide_stratified_folds


def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)


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
    D = D.assign(rt_median_distance = D.rt - D.median_rt,
                 mass_median_distance = D.mass - D.median_mass,
                 dt_median_distance = D.dt - D.median_dt)
    return D


def filter_and_fold(D,
                    D_stats,
                    folds_no=5,
                    fold=peptide_stratified_folds):
    """Get folds for cross-validation.

    Trims the supplied data to subsets that can be folded and 
    assigns to folds.
    """
    D_stats.sort_values("runs", inplace=True)
    run_cnts = D_stats.groupby("runs").runs.count()
    run_cnts = run_cnts[run_cnts >= folds_no].copy()
    D_stats = D_stats.loc[D_stats.runs.isin(run_cnts.index)].copy()
    # we need sorted DF to append a column correctly
    
    D_stats['fold'] = fold(peptides_cnt = len(D_stats),
                           run_cnts = run_cnts,
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
               fold=peptide_stratified_folds):
    """Preprocess the data for fitting cross-validated splines."""
    D_stats = get_stats(D_all, min_runs_no)
    D = get_medians(D_all, D_stats, min_runs_no)
    D_cv, D_stats, run_cnts = filter_and_fold(D, D_stats, folds_no, fold)
    return D_cv, D_stats, run_cnts