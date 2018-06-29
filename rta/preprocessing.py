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



class CV(object):
    def __init__(self, preprocessed_data, folds_no=10):
        self.D = preprocessed_data.D.copy()
        self.var_names = preprocessed_data.var_names
        self.pept_id = preprocessed_data.pept_id
        self.run_name = preprocessed_data.run_name
        self.stats = preprocessed_data.stats.copy()
        self.folds_no = folds_no
        self.run_cnts = self.get_runs_counts()

    def optimize(self, 
                 feature='rt',
                 cores_no=cpu_count()):
        """Find the best set of parameters for the selected feature.

        Args:
            feature (str): the name of the feature to optimize for.

        Returns:
        """
        with Pool(cores_no) as p:
            results = p.starmap(cv_run_param,
                                tasks_run_param(data,
                                                parameters))

    def get_runs_counts(self):
        """Filter for foldable peptides."""
        run_cnts = self.stats.groupby("runs").runs.count()
        run_cnts = run_cnts[run_cnts >= self.folds_no].copy()
        return run_cnts

    def get_fold(self, scheme=)

cv = CV(dp)
cv.run()
cv.run_cnts
