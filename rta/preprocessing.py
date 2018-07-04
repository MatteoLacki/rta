"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)

from rta.read_in_data import big_data
from rta.models.base_model import Model
from rta.models.base_model import predict, fitted, coef, residuals
from rta.pre.processing import preprocess
from rta.cv.folds import stratified_group_folds
from rta.cv.folds import replacement_folds_strata


folds_no = 10
annotated_all, unlabelled_all = big_data()
dp = preprocess(annotated_peptides=annotated_all)


def set_folds(preprocessed_data,
              feature='rt',
              fold=stratified_group_folds,
              folds_no=10,
              shuffle=True):
    """Assign to folds.

    Args:
        preprocessed_data (pandas.DataFrame): data to assign folds to.
        feature (string):   the name of the feature in the column space of the preprocessed_data that will be aligned.
        fold (function):    the folding function.
        folds_no (int):     the number of folds to split the data into.
        shuffle (boolean):  shuffle the points while folding?
    """
    dp = preprocessed_data
    dp.filter_unfoldable_strata(folds_no)
    if fold.__name__ == 'stratified_group_folds':
        # we want the result to be sorted w.r.t. median rt.
        dp.stats.sort_values(["runs", dp.stat_name + '_' + feature],
                             inplace=True)
    dp.stats['fold'] = fold(dp.strata_cnts, folds_no, shuffle)
    dp.D.drop(labels  = [c for c in dp.D.columns if 'fold' in c], 
              axis    = 1,
              inplace = True)
    dp.D = pd.merge(dp.D, dp.stats[['fold']],
                    left_on='id', right_index=True)
    return dp


#TODO: this part should have the possibility to change the fitting procedure.
# So as to make it possible to refit the data.
def calibrate(preprocessed_data,
              parameters=None,
              feature='rt',
              run='run',
              folds_no=10,
              cores_no=cpu_count(),
              _cv_run_args=[]):
    """Calibrate the model for a given feature.

    Args:
        preprocessed_data (pandas.DataFrame): data including folds.
        parameters (iterable): parameters for the individual run models.
        folds_no (int): number of folds to test the model's generalization capabilities.

    Returns:
        a list of models fitted to runs.
    """
    dp = preprocessed_data

    if not parameters:
        parameters = [{"chunks_no": 2**e} for e in range(2,8)]

    # run calibration on runs and parameters
    def tasks_run_param():
        """Iterate over the data runs and fitting parameters."""
        for r, d_r in dp.D.groupby(run):
            d_r = d_r.sort_values(feature)
            d_r = d_r.drop_duplicates(feature)
            for p in parameters:
                out = [r, d_r, p, dp.folds]
                out.extend(_cv_run_args)
                yield out


    with Pool(cores_no) as p:
        results = p.starmap(cv_run_param,
                            tasks_run_param(preprocessed_data,
                                            parameters))
    best_model = select_best_model(results)
    # align the given dimension
    dp['aligned_' + feature_name] = fitted(best_model)
    return dp, results




def AlignData(annotated_peptides,
              features          = ('rt', 'dt'),
              min_runs_no       = 5,
              fold              = stratified_group_folds,
              folds_no          = 10,
              parameters        = {},
              _DataPreprocessor = {},
              _get_stats        = {},
              _set_fold         = {},
              cores_no          = cpu_count()):
    """Align features."""

    dp = preprocess(annotated_peptides, 
                    min_runs_no,
                    _DataPreprocessor,
                    _get_stats)

    calibrated_alignments = {}
    for feature in features:
        dp = set_folds(dp,
                       feature,
                       folds_no = folds_no,
                       **_set_fold)

        dp, calibrated_alignments[feature] = calibrate()
        
        # def tasks_run_param():
        #     """Iterate over the data runs and fitting parameters."""
        #     for run, d_run in dp.D.groupby(run_name):
        #         d_run = d_run.sort_values(feature)
        #         d_run = d_run.drop_duplicates(feature)
        #         for param in feature_param:
        #             out = [run, d_run, param, dp.folds]
        #             out.extend(other_worker_args)
        #             yield out

        # with Pool(cores_no) as p:
        #     fits['feature'] = p.starmap(cv_run_param,
        #                                 tasks_run_param())

    return calibrated_alignments





dp = get_fold(dp,
              folds_no=10,
              feature='rt',
              fold=stratified_group_folds,
              fold_kwds={'shuffle': True})
dp.D






# class RunModel(Model):
#     def __init__(self, preprocessed_data, folds_no=10):
#         self.D = preprocessed_data.D.copy()
#         self.var_names = preprocessed_data.var_names
#         self.pept_id = preprocessed_data.pept_id
#         self.run_name = preprocessed_data.run_name
#         self.stats = preprocessed_data.stats.copy()
#         self.folds_no = folds_no
#         self.run_cnts = self.get_runs_counts()

#     def calibrate(self, 
#                   feature='rt',
#                   cores_no=cpu_count()):
#         """Find the best set of parameters for the selected feature.

#         Args:
#             feature (str): the name of the feature to calibrate.

#         Returns:
#         """
#         with Pool(cores_no) as p:
#             results = p.starmap(cv_run_param,
#                                 tasks_run_param(data,
#                                                 parameters))

#     def get_runs_counts(self):
#         """Filter for foldable peptides."""
#         run_cnts = self.stats.groupby("runs").runs.count()
#         run_cnts = run_cnts[run_cnts >= self.folds_no].copy()
#         return run_cnts

#     def get_fold(self):
#         pass

# rm = RunModel(dp)
# rm.run()
# rm.run_cnts

