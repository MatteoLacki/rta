"""The (deprecated) preprocessing routines."""
import numpy  as np
import pandas as pd

from rta.preprocessing import ordered_str


class DataPreprocessor(object):
    def __init__(self,
                 annotated_peptides,
                 var_names   = ('rt', 'dt', 'mass'),
                 pept_id     = 'id',
                 run_name    = 'run',
                 charge_name = 'charge'):
        """Initialize the DataPreprocessor.

        Args:
            annotated_peptides (pandas.DataFrame): A DataFrame with identified peptides.
            var_names (iter of strings): names of columns for which to obtain the statistic.
            pept_id (str): name of column that identifies peptides in different runs.
        """
        all_col_names = list((pept_id, run_name, charge_name) + var_names)
        assert all(vn in annotated_peptides.columns for vn in all_col_names),\
             "Not all variable names are among the column names of the supplied data frame."
        self.var_names = var_names
        self.run_name  = run_name
        self.pept_id   = pept_id

        # the slimmed data-set: copy is quick and painless.
        self.D = annotated_peptides[all_col_names].copy()
        self.filtered_unfoldable = False
        self.filtered_different_charges_across_runs = False
        # self.calibration_cnt = {vn: 0 for vn in var_names}


    def get_stats(self, stat = np.median,
                        min_runs_no = 5,
                        retain_all_stats = False):
        """Calculate basic statistics conditional on peptide-id.

        Describe the set of peptides appearing in all runs of the experiment.
        'peptide_no' summarizes the distribuant of peptides w.r.t. appearance in runs (starting from those that appear in all runs).
        'self.stats' contain information about the appearance of peptides in different runs.
        'runs' contain the range of values that tag the different runs.

        Args:
            stat (function):   A statistic to apply to the selected features.
            min_runs_no (int): Analyze only peptides that appear at least in that number of runs.
            retain_all_stats (logical): retain general statistics on peptides without focus on those that appear at least in 'min_runs_no'?
        """
        self.min_runs_no = min_runs_no
        self.stat_name   = stat.__name__

        D_id  = self.D.groupby('id')
        stats = D_id[self.var_names].agg(stat)

        stats.columns    = [n + "_" + self.stat_name for n in self.var_names]
        stats['runs_no'] = D_id.id.count()
        # this should be rewritten in C.
        stats['runs']    = D_id.run.agg(ordered_str)

        # counting unique charge states per peptide group
        stats['charges'] = D_id.charge.nunique()
        if retain_all_stats:
            self.all_stats = stats
        runs_no_stats    = stats.groupby('runs_no').size().values
        peptides_no      = np.flip(runs_no_stats, 0).cumsum()
        self.peptides_no = {i + 1:peptides_no[-i-1] for i in range(len(runs_no_stats))}
        self.stats       = stats[stats.runs_no >= self.min_runs_no].copy()
        self.runs        = self.D[self.run_name].unique()


    def get_distances_to_stats(self):
        """Calculate the distances of selected features to their summarizing statistic."""

        # filter the runs that did not appear more than 'min_runs_no'.
        self.D = pd.merge(self.D, self.stats, left_on="id", right_index=True)
        distances = {}
        for n in self.var_names:
            var_stat = n + "_" + self.stat_name
            distances[var_stat + "_distance"] = self.D[n] - self.D[var_stat]
        self.D = self.D.assign(**distances)


    def _trim_stats_and_D(self, retain):
        """Filter peptides that are not in 'self.stats.index'.

        Args:
            retain (np.array of logicals): retain these peptides within 'self.stats'
        """
        self.stats = self.stats[retain].copy()
        self.D = self.D[self.D[self.pept_id].isin(self.stats.index)].copy()


    def filter_unfoldable_strata(self, folds_no):
        """Filter peptides that cannot be folded.

        Trim both stats on peptides and run data of all peptides
        that are not appearing in the same runs in a group of at
        least 'folds_no' other peptides.
        Do it once only per dataset.

        Args:
            folds_no (int): the number of folds to divide the data into.
        """
        if not self.filtered_unfoldable:
            self.folds_no = folds_no
            self.folds    = np.arange(folds_no)
            strata_cnts   = self.stats.groupby("runs").runs.count()
            self.strata_cnts = strata_cnts[strata_cnts >= self.folds_no].copy()
            self._trim_stats_and_D(np.isin(self.stats.runs,
                                           self.strata_cnts.index))
            self.filtered_unfoldable = True


    def filter_multiply_charged(self):
        """Filter peptides that appear in different charge states across different runs of the experiment.

        I still feel that this should not belong here.
        This function will not be used by all alligners.
        """
        if not self.filtered_different_charges_across_runs:
            self._trim_stats_and_D(self.stats.charges == 1)
            self.filtered_different_charges_across_runs = True


# annotated_peptides = annotated_all
# min_runs_no = 5
# var_names   = ('rt', 'dt', 'mass')
# pept_id     = 'id'
# run_name    = 'run'
# charge_name = 'charge'
