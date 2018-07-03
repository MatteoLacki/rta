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
from rta.preprocessing.preprocessing import preprocess


annotated_all, unlabelled_all = big_data()
dp = preprocess(annotated_peptides=annotated_all)


from multiprocessing import Pool, cpu_count
from rta.xvalidation.cross_validation import tasks_run_param, cv_run_param
from rta.xvalidation.stratifications_folds import stratified_folds


class Folder(object):
    def __init__(self, preprocessed_data, folds_no=10):
        self.D = preprocessed_data.D.copy()
        self.var_names = preprocessed_data.var_names
        self.pept_id = preprocessed_data.pept_id
        self.run_name = preprocessed_data.run_name
        self.stats = preprocessed_data.stats.copy()
        self.folds_no = folds_no
        self.get_runs_counts() # this can be used multiple times
        self.filter_foldable() # this too

    def get_runs_counts(self):
        """Filter for foldable peptides."""
        self.runs_cnt = self.stats.groupby("runs").runs.count()

    def filter_foldable(self):
        self.runs_cnt = self.runs_cnt[self.runs_cnt >= self.folds_no].copy()
        self.unique_runs = self.runs_cnt.index
        self.stats = self.stats.loc[self.stats.runs.isin(self.unique_runs)].copy() 

    def fold(self, scheme=stratified_folds):
        pass


def folder():
    


def Fold(preprocessed_data,
         folds_no=10,
         folder):


folder = Folder(preprocessed_data=dp, folds_no=10)






def Solver(problems, 
           solution,
           cores_no=cpu_count()):
    with Pool(cores_no) as p:
        out = p.starmap(cv_run_param,
                        tasks_run_param(data,
                                        parameters))
    return out


# how should it all look like?

# def calibrate(self, 
#               feature='rt',
#               ):
#     """Find the best set of parameters for the selected feature.

#     Args:
#         feature (str): the name of the feature to optimize for.

#     Returns:
#     """
#     assert feature in self.var_names, "Wrong name of the variable."

