"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)
from rta.read_in_data import big_data
from rta.models.base_model import Model
from rta.models.base_model import predict, fitted, coef, residuals
from rta.preprocessing.preprocessing import preprocess

folds_no = 10
annotated_all, unlabelled_all = big_data()
dp = preprocess(annotated_peptides=annotated_all)





from rta.xvalidation.folds import stratified_group_folds
from rta.xvalidation.folds import replacement_folds_strata

preprocessed_data = dp
D = dp.D
stats = dp.stats
run_cnts = dp.run_cnts
dp.stat_name
dp.strata_cnts

def set_fold(preprocessed_data,
             feature='rt',
             fold=stratified_group_folds,
             folds_no=10,
             shuffle=True):
    """Assign to folds."""
    d = preprocessed_data
    d.filter_unfoldable_strata(folds_no)
    if fold.__name__ == 'stratified_group_folds':
        # we want the result to be sorted w.r.t. median rt.
        d.stats.sort_values(["runs", d.stat_name + '_' + feature],
                            inplace=True)
    d.stats['fold'] = fold(d.strata_cnts, folds_no, shuffle)
    d.D.drop(labels  = [c for c in d.D.columns if 'fold' in c], 
             axis    = 1,
             inplace = True)
    d.D = pd.merge(d.D, d.stats[['fold']],
                   left_on='id', right_index=True)
    return d

from multiprocessing import Pool, cpu_count
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param



# def tasks_run_param(data,
#                     parameters,
#                     var_name='rt',
#                     run_name='run',
#                    *other_worker_args):
#     """Iterate over the data runs and fitting parameters."""
#     folds = np.unique(data.fold)
#     for run, d_run in data.groupby(run_name):
#         d_run = d_run.sort_values(var_name)
#         d_run = d_run.drop_duplicates(var_name)
#         for param in parameters:
#             out = [run, d_run, param, folds]
#             out.extend(other_worker_args)
#             yield out

# How this should look like?
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

    fits = {}
    for feature in features:
        dp = set_fold(dp,
                      feature,
                      folds_no = folds_no,
                      **_set_fold)

        try:
            feature_param = parameters[feature]
        except KeyError:
            print("Need default.")

        # calibration
        def tasks_run_param():
            """Iterate over the data runs and fitting parameters."""
            for run, d_run in dp.D.groupby(run_name):
                d_run = d_run.sort_values(feature)
                d_run = d_run.drop_duplicates(feature)
                for param in feature_param:
                    out = [run, d_run, param, dp.folds]
                    out.extend(other_worker_args)
                    yield out

        with Pool(cores_no) as p:
            fits['feature'] = p.starmap(cv_run_param,
                                        tasks_run_param())







dp = get_fold(dp,
              folds_no=10,
              feature='rt',
              fold=stratified_group_folds,
              fold_kwds={'shuffle': True})
dp.D





def calibrate(preprocessed_data,
              parameters,
              feature_name='rt',
              run_name='run',
              folds_no=10,
              cores_no=cpu_count(),
              *other_worker_args):
    """Calibrate the model for a given feature.

    Args:
        preprocessed_data (pandas.DataFrame): data to work on.
        parameters (iterable): parameters for the individual run models.
        folds_no (int): number of folds to test the model's generalization capabilities.

    Returns:
        a list of models fitted to runs.
    """

    # get folds


    # run calibration on runs and parameters
    def tasks_run_param():
        """Iterate over the data runs and fitting parameters."""
        folds = np.unique(preprocessed_data.fold)
        for run, d_run in data.groupby(run_name):
            d_run = d_run.sort_values(var_name)
            d_run = d_run.drop_duplicates(var_name)
            for param in parameters:
                out = [run, d_run, param, folds]
                out.extend(other_worker_args)
                yield out

    with Pool(cores_no) as p:
        results = p.starmap(cv_run_param,
                            tasks_run_param(preprocessed_data,
                                            parameters))
    return results




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
