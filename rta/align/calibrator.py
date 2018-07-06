"""A class for calibrating the parameters of the model. """

from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

from rta.cv.cv              import cv_run_param
from rta.cv.folds           import replacement_folds_strata
from rta.cv.folds           import stratified_group_folds
from rta.models.base_model  import fitted, predict
from rta.models.robust_spline import RobustSpline
from rta.stats.stats        import mae, mad, confusion_matrix




def cv_run_param(run_no,
                 d_run,
                 parameter,
                 folds,
                 feature,
                 feature_stat,
                 Model=RobustSpline,
                 fold_stats=(mae, mad),
                 model_stats=(np.mean, np.median, np.std)):
    """Cross-validate a model under a given 'run' and 'parameter'."""
    m = Model()
    m.fit(d_run[feature].values, 
          d_run[feature_stat].values,
          **parameter)
    m_stats = []
    cv_out = []
    for fold in folds:
        train = d_run.loc[d_run.fold != fold,:]
        test  = d_run.loc[d_run.fold == fold,:]
        n = Model()
        n.fit(x=train[feature].values,
              y=train[feature_stat].values,
              **parameter)
        errors = np.abs(predict(n, test[feature].values) - \
                        test[feature_stat].values)
        n_signal = n.is_signal(test.rt, test.rt_median_distance)
        stats = [stat(errors) for stat in fold_stats]
        m_stats.append(stats)
        cm = confusion_matrix(m.signal[d_run.fold == fold], n_signal)
        cv_out.append((n, stats, cm))
    # process stats
    m_stats = np.array(m_stats)
    m_stats = np.array([stat(m_stats, axis=0) for stat in model_stats])
    m_stats = pd.DataFrame(m_stats)
    m_stats.columns = ["fold_" + fs.__name__ for fs in fold_stats]
    m_stats.index = [ms.__name__ for ms in model_stats]

    return run_no, parameter, m, m_stats, cv_out



class Calibrator(object):
    def __init__(self, 
                 preprocessed_data,
                 feature='rt',
                 folds_no=10,
                 run='run'):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pandas.DataFrame): data to assign folds to.
            feature (string):   the name of the feature in the column space of the preprocessed_data that will be aligned.
        """
        
        # filter out peptides that occur in runs in groups smaller than ''
        preprocessed_data.filter_unfoldable_strata(folds_no)
        self.D = preprocessed_data.D
        self.stats = preprocessed_data.stats
        self.feature = feature
        self.feature_stat = feature + '_' + preprocessed_data.stat_name
        self.feature_stat_distance = feature_stat + '_distance'
        self.run = run

    def set_folds(self,
                  folds_no=10,
                  fold=stratified_group_folds,
                  shuffle=True):
        """Assign to folds.

        Args:
            fold (function):    the folding function.
            folds_no (int):     the number of folds to split the data into.
            shuffle (boolean):  shuffle the points while folding?
        """
        # TODO: this should really be some wrapper around the silly method.
        self.folds_no = folds_no # write a setter to check for proper type and value

        if fold.__name__ == 'stratified_group_folds':
            # we want the result to be sorted w.r.t. median rt.
            self.stats.sort_values(["runs", self.feature_stat], inplace=True)

        # get fold assignments for pept-id groups
        self.stats['fold'] = fold(self.d.strata_cnts,
                                     self.folds_no,
                                     shuffle)

        # if there was some other fold in the dataframe - drop it!
        self.d.D.drop(labels = [c for c in self.D.columns if 'fold' in c], 
                       axis = 1,
                       inplace = True)

        # propage fold assignments to the main data
        self.D = pd.merge(self.D,
                          self.stats[['fold']],
                          left_on='id',
                          right_index=True)

    def iter_run_param(self):
        """Iterate over the data runs and fitting parameters."""
        # iterate over runs
        for r, d_r in self.D.groupby(self.run):
            #TODO: what to do with these values?
            d_r = d_r.sort_values(self.feature)
            # only one procedure requires the values to be 
            # ordered and without duplicates...
            d_r = d_r.drop_duplicates(self.feature)
            for p in self.parameters:
                # TODO run and other info should go into _cv_run_args
                out = [r, d_r, p,
                       self.folds,
                       self.feature,
                       self.feature_stat]
                out.extend(self._cv_run_args)
                yield out

    def select_best_model(self):
        """Select the best model from the results."""
        pass


    # definately, the cv should be part of the model.
    def cv_run_param(self,
                     run_no,
                     d_run,
                     parameter,
                     Model=RobustSpline,
                     fold_stats=(mae, mad),
                     model_stats=(np.mean, np.median, np.std)):
        """Cross-validate a model under a given 'run' and 'parameter'."""
        m = Model()
        m.fit(d_run[self.feature].values, 
              d_run[self.feature_stat].values,
              **parameter)
        m_stats = []
        cv_out = []
        for fold in self.folds:
            train = d_run.loc[d_run.fold != fold,:]
            test  = d_run.loc[d_run.fold == fold,:]
            n = Model()
            n.fit(x=train[self.feature].values,
                  y=train[self.feature_stat].values,
                  **parameter)
            errors = np.abs(predict(n, test[self.feature].values) - \
                            test[self.feature_stat].values)
            n_signal = n.is_signal(test.rt, test.rt_median_distance)
            stats = [stat(errors) for stat in fold_stats]
            m_stats.append(stats)
            cm = confusion_matrix(m.signal[d_run.fold == fold], n_signal)
            cv_out.append((n, stats, cm))
        # process stats
        m_stats = np.array(m_stats)
        m_stats = np.array([stat(m_stats, axis=0) for stat in model_stats])
        m_stats = pd.DataFrame(m_stats)
        m_stats.columns = ["fold_" + fs.__name__ for fs in fold_stats]
        m_stats.index = [ms.__name__ for ms in model_stats]

        return run_no, parameter, m, m_stats, cv_out


    def calibrate(self,
                  parameters=None,
                  cores_no=cpu_count(),
                  _cv_run_args=[]):
        if not parameters:
            parameters = [{"chunks_no": 2**e} for e in range(2,8)]
        self.parameters = parameters
        self._cv_run_args = _cv_run_args

        with Pool(cores_no) as p:
            self.results = p.starmap(cv_run_param,
                                     self.iter_run_param())

        self.select_best_model()
        # align the given dimension
        self.d['aligned_' + self.feature] = fitted(best_model)



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

