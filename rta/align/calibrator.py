"""A class for calibrating the parameters of the model. """

import matplotlib.pyplot  as plt
from collections      import defaultdict
from math             import inf
from multiprocessing  import Pool, cpu_count
import numpy              as np
import pandas             as pd

from rta.array_operations.misc import iter_cluster_ends
from rta.cv.folds              import replacement_folds_strata
from rta.cv.folds              import stratified_group_folds
from rta.models.base_model     import fitted, predict
from rta.models.splines.robust import robust_spline


# TODO: this should support different models.
def cv_run_param_old(r, x, y, f, p):
    """Little wrapper around the robust splines.

    Args:
        r (int or str): The number of given run.
        x (np.array):   Control (usually retention time or drift time).
        y (np.array):   Response (usually distance to the median retention time or drift time).
        f (np.array):   Assignments to different folds.
        p (dict):       A dictionary of additional parameters for the robust spline.
    """
    m = robust_spline(x, y,
                      drop_duplicates = False,
                      sort            = False,
                      folds           = f,
                      **p)
    return r, p, m




def cv_run_param(r, x, y, m, f, p):
    """Little wrapper around the robust splines.

    Args:
        r (int or str): The number of given run.
        x (np.array):   Control (usually retention time or drift time).
        y (np.array):   Response (usually distance to the median retention time or drift time).
        f (np.array):   Assignments to different folds.
        p (dict):       A dictionary of additional parameters for the robust spline.
    """
    fm = m(x, y,
           drop_duplicates = False,
           sort            = False,
           folds           = f,
           **p)
    return r, p, fm



class Calibrator(object):
    def __init__(self, 
                 data,
                 feature='rt',
                 folds_no=10):
        """Initialize the Calibrator.

        Args:
            data (pandas.DataFrame): data to assign folds to.
            feature (string): the name of the feature in the column space of the data that will be aligned.
            folds_no (int):     the number of folds to split the data into.
        """
        self.folds_no = folds_no
        self.stats    = data['stats'].copy()
        self.d        = data
        # setting alignment indices.
        self.align_it = 0
        self.stat_col = 'runs_stat_0'
        self.feature  = feature
        self.y        = 'runs_stat_dist_0'
        self.x        = feature + "_0" # self.y will store current name of the response variable.
        # local copy of the dataset, trimmed to necessary columns.
        # this is where additional columns with aligned values will appear?
        self.D        = self.d['data'][[self.d['pept_id' ],
                                        self.d['run_name'],
                                        feature]]
        self.D.columns = ['id', 'run', self.x]
        self.best_models = {}


    def increase_align_cnt(self):
        self.align_it += 1
        self.stat_col = 'runs_stat_{}'.format(self.align_it)
        self.y        = 'runs_stat_dist_{}'.format(self.align_it)
        self.x        = self.feature + "_{}".format(self.align_it)


    def runs_statistic(self, stat = np.median):
        """Compute the statistic for each peptide group and compute the distance to it.

        Distance to this statistic will be the basis for the alignment.
        We append it to the local copy of the aligned data, D, too.

        Args:
            stat (function): the statistic to be applied.
        """
        # let align_it store always the most recent alignment.
        peptide_grouped_feature = self.D.groupby(self.d['pept_id'])[self.x]
        # the current name of the summary statistic in the stats
        self.stats[self.stat_col] = peptide_grouped_feature.agg(stat)
        self.D = pd.merge(self.D,
                          self.stats[[self.stat_col]],
                          left_on     = "id",
                          right_index = True)
        self.D[self.y] = self.D[self.x] - self.D[self.stat_col]


    def fold(self,
             fold    = stratified_group_folds,
             shuffle = True):
        """Assign to folds.

        Args:
            fold (function):   the folding function.
            shuffle (boolean): shuffle the points while folding?
        """
        if fold.__name__ == 'stratified_group_folds':
            # we want the result to be sorted w.r.t. the applied statistic
            # because for naive sortings we assume that the order of appearance
            # of peptides is 'stochastic'.
            # If shuffle == True, this will not be of any importance.
            self.stats.sort_values(["runs", self.stat_col], inplace=True)
        # get fold assignments for pept-id groups.
        # strata_cnts contains info on the count of appearance of peptides in runs.
        # for instance: 1_2_3_4_10 -> 48
        # 'fold' will simply produce 48 assignments for this row.
        # this will be followed by assignments for the next group, e.g. 1_2_3_4_5.
        # that's why we need to sort by 'runs'.
        # alternatively, we would have to create an assignment per each group.
        # but I don't know Pandas too good to do this that way...
        self.stats['fold'] = fold(self.d['strata_cnts'],
                                  self.folds_no,
                                  shuffle)

        # propagate fold assignments to the main data
        self.D = pd.merge(self.D,
                          self.stats[['fold']],
                          left_on='id',
                          right_index=True)


    def iter_run_param(self, 
                       sort            = True,
                       drop_duplicates = True,
                       model           = robust_spline):
        """Iterate over the data runs and fitting parameters.

        Args:
            sort (logical):            sort the data by the control variable 'x'?
            drop_duplicates (logical): drop duplicates in the control variable 'x'?
        """
        folds = np.arange(self.folds_no)
        for r, d_r in self.D.groupby('run'):
            if sort:
                d_r = d_r.sort_values(self.x)
            if drop_duplicates:
                d_r = d_r.drop_duplicates(self.x)
            for p in self.parameters:
                yield (r, d_r[self.x].values,
                          d_r[self.y].values,
                          model,
                          d_r.fold.values, p)


    def calibrate(self,
                  parameters = None,
                  model      = robust_spline,
                  cores_no   = cpu_count()):
        """Calibrate the selected dimension of the data.

        Args:
            parameters (iterable):  Dictionaries with parameter-value entries for the model.
            cores_no (int):         Number of cores use by multiprocessing.
        """
        if not parameters:
            parameters = [{"chunks_no": 2**e} for e in range(2,10)]
        self.parameters = parameters

        with Pool(cores_no) as p:
            self.cal_res = p.starmap(cv_run_param,
                                     self.iter_run_param())


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
        self.best_models[self.align_it] = best_models


    def align(self):
        """Align self.D for calibration."""
        self.D  = self.D.sort_values('run')
        feature = self.D[self.x].values
        outcome = np.full(shape = feature.shape,
                          dtype = self.D[self.x].dtype,
                          fill_value = 0)
        for s, e, r in iter_cluster_ends(np.nditer(self.D.run)):
            X = feature[s:e]
            # we model X - Stat(X) = F(X) + err. Next iteration, is nothing else, but 
            # X - F(X).
            outcome[s:e] = X - self.best_models[self.align_it][r](X)
        self.increase_align_cnt()
        self.D[self.x] = outcome


    def __call__(self, x, run):
        """Align the data from a given run.

        Args:
            x (np.array): Points to align.
            run (int):    Number of the run to align.
        Returns:
            np.array :    Aligned points.
        """
        for i in range(self.align_it):
            # application of consecutive models
            x = x - self.best_models[i][run](x)
        return x


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
        for r in self.d['runs']:
            x, y = opt_var_vals, stats[r]
            plt.semilogx(x, y, basex=2, label=r)
            plt.text(x[ 0], y[ 0], 'Run {}'.format(r))
            plt.text(x[-1], y[-1], 'Run {}'.format(r))
        plt.xlabel(opt_var)
        plt.ylabel(y_label)
        plt.title('Training-Test comparison')
        if show:
            plt.show()


    def plot_run_alignments(self,
                            run_tag,
                            max_alignments = inf,
                            plt_style      = 'dark_background',
                            show           = True,
                          **kwds):
        """Plot the alignments.

        Args:
            run_tag (int):   The tag of the run to plot.
            plt_style (str): Matplotlib style to apply to the plot. This probably should be a global setting.
            show (logical):  Should we show the plot immediately, or do you want to add some more things before running "plt.show"?
        """
        plt.style.use(plt_style)
        max_plots  = min(self.align_it, max_alignments)
        first_plot = plt.subplot(max_plots, 1, 1, 
                                 title = "Alignment No 1")
        self.best_models[0][run_tag].plot(show = False, **kwds)
        for i in range(1, max_plots):
            plt.subplot(max_plots, 1, i+1,
                        sharex = first_plot,
                        sharey = first_plot,
                        title  = "Alignment No {}".format(i+1))
            self.best_models[i][run_tag].plot(show = False, **kwds)
        if show:
            plt.show()


# these functions should be updated to make a repertoire of methods.
def calibrate(feature,
              data,
              folds_no,
              min_runs_no,
              align_cnt  = 2,
              runs_stat  = np.median,
              parameters = None,
              model      = robust_spline,
              cores_no   = cpu_count()):
    """Calibrate the given feature.

    Args:
        folds_no (int):     The number of folds to split the data into.
        min_runs_no (int):
        data (pd.DataFrame):
        feature (string): The name of the feature in the column space of the data that will be aligned.
        align_cnt (int):  The number of times that the feature is aligned.
    """
    c = Calibrator(data, feature = feature, folds_no = folds_no)
    c.runs_statistic(stat = runs_stat)
    c.fold()
    for _ in range(align_cnt):
        c.calibrate(parameters = None,
                    model      = robust_spline,
                    cores_no   = cpu_count())
        c.select_best_models()
        c.align()
        c.runs_statistic(stat = runs_stat)
    return c
