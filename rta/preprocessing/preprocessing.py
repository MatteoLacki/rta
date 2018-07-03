import numpy as np
import pandas as pd


#TODO: replace this with C.
def ordered_str(x):
    x = x.values
    x.sort()
    return "_".join(str(i) for i in x)



class DataPreprocessor(object):
    def __init__(self,
                 annotated_peptides,
                 var_names=('rt', 'dt', 'mass'),
                 pept_id='id',
                 run_name='run'):
        """Initialize the DataPreprocessor.

        Args:
            annotated_peptides (pandas.DataFrame): A DataFrame with identified peptides.
            var_names (iter of strings): names of columns for which to obtain the statistic.
            pept_id (str): name of column that identifies peptides in different runs.
        """
        all_col_names = list((pept_id,run_name) + var_names)
        assert all(vn in annotated_peptides.columns for vn in all_col_names),\
             "Not all variable names are among the column names of the supplied data frame."
        self.var_names = var_names
        self.run_name  = run_name
        self.pept_id   = pept_id        
        # the slimmed data-set: copy is quick and painless.
        self.D = annotated_peptides[all_col_names].copy()

    def get_stats(self, stat=np.median, min_runs_no=5):
        """Calculate basic statistics conditional on peptide-id.

        Args:
            D_all (pandas.DataFrame): The DataFrame with identified peptides.
            min_runs_no (int): Analyze only peptides that appear at least in that number of runs.
            var_names (iter of strings): names of columns for which to obtain the statistic.
            pept_id (str): name of column that identifies peptides in different runs.
            stat (function): a statistic to apply to the selected features.

        Return:
            D_stats (pandas.DataFrame): A DataFrame summarizing the selected features of peptides. Filtered for peptides appearing at least in a given minimal number of runs. 

        """
        self.min_runs_no = min_runs_no
        self.stat_name = stat.__name__
        D_id = self.D.groupby('id')
        stats = D_id[self.var_names].agg(stat)
        stats.columns = [self.stat_name + "_" + vn for vn in self.var_names]
        stats['runs_no'] = D_id.id.count()
        # this should be rewritten in C.
        stats['runs'] = D_id.run.agg(ordered_str)
        self.stats = stats[stats.runs_no >= self.min_runs_no].copy()

    def get_distances_to_stats(self):
        """Calculate the distances of selected features to their summarizing statistic."""

        # filter the runs that did not appear more than 'min_runs_no'.
        self.D = pd.merge(self.D, self.stats, left_on="id", right_index=True)
        distances = {}
        for n in self.var_names:
            var_stat = self.stat_name + "_" + n
            distances[var_stat + "_distance"] = self.D[n] - self.D[var_stat]
        self.D = self.D.assign(**distances)

    def filter_unfoldable_strata(self, folds_no):
        """Filter peptides that cannot be folded.

        Trim both stats on peptides and run data of all peptides
        that are not appearing in the same runs in a group of at
        least 'folds_no' other peptides."""
        self.folds_no = folds_no
        run_cnts      = self.stats.groupby("runs").runs.count()
        self.run_cnts = run_cnts[run_cnts >= self.folds_no].copy()
        # filtering stats
        self.stats    = self.stats[np.isin(self.stats.runs,
                                           self.run_cnts.index)].copy()
        # filtering data
        self.D = self.D[self.D[self.pept_id].isin(self.stats.index)].copy()


def preprocess(annotated_peptides,
               min_runs_no=5,
               _DataPreprocessor={},
               _get_stats={}):
    """Wrapper around preprocessing of the annotated peptides.""" 
    dp = DataPreprocessor(annotated_peptides, **_DataPreprocessor)
    dp.get_stats(min_runs_no=5, **_get_stats)
    dp.get_distances_to_stats()
    return dp
