"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2


import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 4)

from rta.align.calibrator import Calibrator
from rta.read_in_data import big_data
from rta.preprocessing import preprocess

# from rta.models.base_model import Model
# from rta.models.base_model import predict, fitted, coef, residuals
from rta.cv.folds   import stratified_group_folds
# from rta.cv.folds import replacement_folds_strata

if __name__ == "__main__":
    folds_no = 10
    annotated_all, unlabelled_all = big_data()

    d = preprocess(annotated_peptides=annotated_all)
    d.keys()
    d['data']
    d['stats'].columns
    feature = 'rt'

    calibrator = Calibrator(d, feature)
    calibrator.fold(folds_no, stratified_group_folds, True)


    from rta.models.SQSpline import SQSpline

    calibrator.d.D


    model = SQSpline()
    model.fit()

    calibrator.parameters = [{"chunks_no": 2**e} for e in range(2,8)]
    calibrator._cv_run_args = []
    calibrator.calibrate()


    it = calibrator.iter_run_param()
    next(it)


    from rta.align.calibrator import cv_run_param


    cv_run_param(*next(it))
    run_no, d_run, parameter, folds, feature, feature_stat = next(it)



    # tests for what the methods should be returning.



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
            results = p.starmap(cv_run_param, tasks_run_param())
        best_model = select_best_model(results)
        # align the given dimension
        dp['aligned_'+feature_name] = fitted(best_model)
        return dp, results


    def align(annotated_peptides,
              features          = ('rt', 'dt'),
              min_runs_no       = 5,
              fold              = stratified_group_folds,
              folds_no          = 10,
              parameters        = {},
              _DataPreprocessor = {},
              _get_stats        = {},
              _set_fold         = {}):
        """Align features."""

        # preprocessing common to all the features under alignment
        dp = preprocess(annotated_peptides, 
                        min_runs_no,
                        _DataPreprocessor,
                        _get_stats)

        calibrated_alignments = {}
        for feature in features:
            # put set_folds to init of the calibrator class?
            dp = set_folds(dp,
                           feature,
                           folds_no = folds_no,
                           **_set_fold)

            dp, calibrated_alignments[feature] = calibrate()

        return calibrated_alignments


