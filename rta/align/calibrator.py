"""A class for calibrating the parameters of the model. """

import matplotlib.pyplot as plt
from collections      import defaultdict
from math             import inf
from multiprocessing  import Pool, cpu_count
import numpy  as np
import pandas as pd

from rta.cv.folds              import replacement_folds_strata
from rta.cv.folds              import stratified_group_folds
from rta.models.base_model     import fitted, predict
from rta.models.splines.robust import robust_spline



def cv_run_param(r, x, y, f, p):
    m = robust_spline(x, y,
                      drop_duplicates = False,
                      sort            = False,
                      folds           = f,
                      **p)
    return r, p, m


class Calibrator(object):
    def __init__(self, 
                 preprocessed_data,
                 feature='rt',
                 folds_no=10):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pandas.DataFrame): data to assign folds to.
            feature (string): the name of the feature in the column space of the preprocessed_data that will be aligned.

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
             fold    = stratified_group_folds,
             shuffle = True):
        """Assign to folds.

        Args:
            fold (function):    the folding function.
            folds_no (int):     the number of folds to split the data into.
            shuffle (boolean):  shuffle the points while folding?
        """
        if fold.__name__ == 'stratified_group_folds':
            # we want the result to be sorted w.r.t. the applied statistic
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
                       sort            = True,
                       drop_duplicates = True):
        """Iterate over the data runs and fitting parameters.

        Args:
            sort (logical):             sort the data by the control variable 'x'?
            drop_duplicates (logical):  drop duplicates in the control variable 'x'?

        """
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


    def select_best_models(self,
                           fold_stat = 'fold_mad',
                           stat      = 'median'):
        """Select best models for each run.

        Args:
            fold_stat (str): Name of the statistic used to summarize errors on the test set on one fold.
            stat (str):      Name of the statistic used to summarize fold statistics.
        Returns:
            
        """
        bm_stat_min = defaultdict(lambda: inf)
        best_models = {}

        for r, p, m in self.cal_res:
            min_stat = m.cv_stats.loc[stat, fold_stat]
            if min_stat < bm_stat_min[r]:
                bm_stat_min[r] = min_stat
                best_models[r] = m
        self.best_models = best_models


    def calibrate(self,
                  parameters = None,
                  cores_no   = cpu_count()):
        """Calibrate the selected dimension of the data.

        Args:
            parameters (iterable):  Dictionaries with parameter-value entries.
            cores_no (int):         Number of cores use by multiprocessing.
        """
        if not parameters:
            parameters = [{"chunks_no": 2**e} for e in range(2,10)]
        self.parameters = parameters

        with Pool(cores_no) as p:
            self.cal_res = p.starmap(cv_run_param,
                                     self.iter_run_param())

    def plot(self,
             opt_var   = 'chunks_no',
             fold_stat = 'fold_mad',
             stat      = 'median',
             y_label   = 'median fold median absolute error',
             plt_style = 'dark_background',
             show      = True,
           **kwds):
        """Plot calibration error curves.

        Args:
            opt_var (str):   The name of the variable to optimize.
            fold_stat (str): Name of the statistic used to summarize errors on the test set on one fold.
            stat (str):      Name of the statistic used to summarize fold statistics.
            plt_style (str): Matplotlib style to apply to the plot. This probably should be a global setting.
            show (logical):  Should we show the plot immediately, or do you want to add some more things before running "plt.show"?
        """
        plt.style.use(plt_style)
        opt_var_vals = sorted([p[opt_var] for p in self.parameters])
        stats = defaultdict(list)
        # mad_std  = defaultdict(list)
        for r, p, m in self.cal_res:
            s = m.cv_stats
            stats[r].append(s.loc[stat, fold_stat])
            # mad_std[r].append( s.loc['std',  'fold_mae'])

        for r in self.d.runs:
            x, y = opt_var_vals, stats[r]
            plt.semilogx(x, y, basex=2, label=r)
            plt.text(x[ 0], y[ 0], 'Run {}'.format(r))
            plt.text(x[-1], y[-1], 'Run {}'.format(r))

        plt.xlabel(opt_var)
        plt.ylabel(y_label)
        plt.title('Training-Test comparison')

        if show:
            plt.show()



class DTcalibrator(Calibrator):
    def __init__(self,
                 preprocessed_data,
                 feature='dt',
                 folds_no=10):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pd.DataFrame): data to assign folds to.
            feature (string): the name of the feature in the column space of the preprocessed_data that will be aligned.

        """
        self.folds_no = folds_no
        self.d = preprocessed_data
        self.d.filter_unfoldable_strata(self.folds_no)
        self.feature = feature
        self.feature_stat = feature + '_' + preprocessed_data.stat_name
        self.feature_stat_distance = self.feature_stat + '_distance'
        self.D = self.d.D
        self.pept_id = self.d.pept_id
        self.stats = self.d.stats
        self._trim_stats_and_D(self.stats.charges == 1)
        self.D = self.D[[self.d.pept_id,
                         self.d.run_name,
                         self.feature,
                         self.feature_stat_distance]]
        self.D.columns = ['id', 'run', 'x', 'y']

    def _trim_stats_and_D(self, retain):
        """Filter peptides that are not in 'self.stats.index'.

        Args:
            retain (np.array of logicals): retain these peptides within 'self.stats'
        """
        self.stats = self.stats[retain].copy()
        self.D = self.D[self.D[self.pept_id].isin(self.stats.index)].copy()



def calibrate_rt(preprocessed_data,
                 folds_no   = 10,
                 fold       = stratified_group_folds,
                 shuffle    = True,
                 parameters = None,
                 cores_no   = cpu_count(),
                 fold_stat = 'fold_mad',
                 stat      = 'median'):
    """Calibrate the retention time.

    Args:
        preprocessed_data (pd.DataFrame): Data to assign folds to.
        folds_no (int):                   Number of folds to split the data into.
        fold (function):                  The function producing folds.
        shuffle (boolean):                Shuffle the points while folding?
        parameters (iterable):            Dictionaries with parameter-value entries.
        cores_no (int):                   Number of cores use by multiprocessing.
        fold_stat (str):                  Name of the statistic used to summarize errors on the test set on one fold.
        stat (str):                       Name of the statistic used to summarize fold statistics.
    Returns:
        Calibrator : a funky instance of a calibrator.
    """
    c = Calibrator(preprocessed_data, feature = 'rt')
    c.set_folds(folds_no, fold, shuffle)
    c.calibrate(parameters, cores_no)
    c.select_best_models(fold_stat, stat)
    return c



def calibrate_dt(preprocessed_data,
                 folds_no   = 10,
                 fold       = stratified_group_folds,
                 shuffle    = True,
                 parameters = None,
                 cores_no   = cpu_count(),
                 fold_stat = 'fold_mad',
                 stat      = 'median'):
    """Calibrate the drift time.

    Args:
        preprocessed_data (pd.DataFrame): Data to assign folds to.
        folds_no (int):                   Number of folds to split the data into.
        fold (function):                  The function producing folds.
        shuffle (boolean):                Shuffle the points while folding?
        parameters (iterable):            Dictionaries with parameter-value entries.
        cores_no (int):                   Number of cores use by multiprocessing.
        fold_stat (str):                  Name of the statistic used to summarize errors on the test set on one fold.
        stat (str):                       Name of the statistic used to summarize fold statistics.
    Returns:
        Calibrator : a funky instance of a calibrator.
    """
    c = Calibrator(preprocessed_data, feature = 'dt')
    c.set_folds(folds_no, fold, shuffle)
    c.calibrate(parameters, cores_no)
    c.select_best_models(fold_stat, stat)
    return c
