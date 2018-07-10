"""A class for calibrating the parameters of the model. """

from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

from rta.cv.folds           import replacement_folds_strata
from rta.cv.folds           import stratified_group_folds
from rta.models.base_model  import fitted, predict
from rta.models.splines.robust import robust_spline



def cv_run_param(r, x, y, f, p):
    m = robust_spline(x, y,
                      drop_duplicates_and_sort=False,
                      folds = f,
                      **p)
    return m



class Calibrator(object):
    def __init__(self, 
                 preprocessed_data,
                 feature='rt',
                 folds_no=10):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pandas.DataFrame): data to assign folds to.
            feature (string):   the name of the feature in the column space of the preprocessed_data that will be aligned.
        """
        
        # filter out peptides that occur in runs in groups smaller than ''
        self.folds_no = folds_no
        self.d = preprocessed_data
        self.d.filter_unfoldable_strata(self.folds_no)
        self.feature = feature
        self.feature_stat = feature + '_' + preprocessed_data.stat_name
        self.feature_stat_distance = self.feature_stat + '_distance'
        self.D = self.d.D[[self.d.pept_id,
                           self.d.run_name,
                           self.feature,
                           self.feature_stat_distance]]
        self.D.columns = ['id', 'run', 'x', 'y']

    def fold(self,
             fold=stratified_group_folds,
             shuffle=True):
        """Assign to folds.

        Args:
            fold (function):    the folding function.
            folds_no (int):     the number of folds to split the data into.
            shuffle (boolean):  shuffle the points while folding?
        """
        if fold.__name__ == 'stratified_group_folds':
            # we want the result to be sorted w.r.t. median rt.
            self.d.stats.sort_values(["runs", self.feature_stat], inplace=True)

        # get fold assignments for pept-id groups
        self.d.stats['fold'] = fold(self.d.strata_cnts,
                                    self.folds_no,
                                    shuffle)

        # propagate fold assignments to the main data
        self.D = pd.merge(self.D,
                          self.d.stats[['fold']],
                          left_on='id',
                          right_index=True)

    def iter_run_param(self, 
                       sort=True,
                       drop_duplicates=True):
        """Iterate over the data runs and fitting parameters."""
        # iterate over runs
        folds = np.arange(self.folds_no)
        for r, d_r in self.D.groupby('run'):
            if sort:
                d_r = d_r.sort_values('x')
            if drop_duplicates:
                d_r = d_r.drop_duplicates('x')
            for p in self.parameters:
                yield (r, d_r.x.values,
                          d_r.y.values,
                          d_r.fold.values, p)

    def select_best_model(self):
        """Select the best model from the results."""
        pass

    def calibrate(self,
                  parameters=None,
                  cores_no=cpu_count()):
        if not parameters:
            parameters = [{"chunks_no": 2**e} for e in range(2,8)]
        self.parameters = parameters

        with Pool(cores_no) as p:
            self.results = p.starmap(cv_run_param,
                                     self.iter_run_param())

    #     self.select_best_model()
    #     # align the given dimension
    #     self.d['aligned_' + self.feature] = fitted(best_model)


def calibrate(preprocessed_data,
              feature='rt',
              folds_no=10,
              fold=stratified_group_folds,
              shuffle=True):
    """Calibrate the given feature of the data."""
    self.feature = feature
    calibrator = Calibrator(preprocessed_data, self.feature)
    calibrator.set_folds(folds_no, fold, shuffle)
    calibrator.calibrate()
    return calibrator.dp, calibrator.results
    # return calibrator.dp, calibrator.best_model

