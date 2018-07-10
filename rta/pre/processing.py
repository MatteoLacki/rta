import numpy as np
import pandas as pd

#TODO: this whole file should be in the main folder.
# there will never be anything like 'pre'


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
        self.filtered_unfoldable = False

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
        stats.columns = [n + "_" + self.stat_name for n in self.var_names]
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
            var_stat = n + "_" + self.stat_name
            distances[var_stat + "_distance"] = self.D[n] - self.D[var_stat]
        self.D = self.D.assign(**distances)

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
            self.folds = np.arange(folds_no)
            strata_cnts = self.stats.groupby("runs").runs.count()
            self.strata_cnts = strata_cnts[strata_cnts >= self.folds_no].copy()
            # filtering stats
            self.stats = self.stats[np.isin(self.stats.runs,
                                    self.strata_cnts.index)].copy()
            # filtering data
            self.D = self.D[self.D[self.pept_id].isin(self.stats.index)].copy()
            self.filtered_unfoldable = True

    # def fold(self, folds_no,
    #                feature='rt',
    #                fold=stratified_folds,
    #                fold_kwds={'shuffle': True}):
    #     self.folds_no = folds_no
    #     # no sense to make foldable unless folds are prepared too
    #     self.__filter_unfoldable_strata()
    #     if fold.__name__ == 'stratified_folds':
    #         # we want the result to be sorted w.r.t. median rt.
    #         sort_vars = ["runs", self.stat_name + '_' + feature]
    #         self.stats.sort_values(sort_vars, inplace=True)
    #     self.stats['fold'] = fold(self.strata_cnts,
    #                               self.folds_no,
    #                               **fold_kwds)
    #     self.D = pd.merge(self.D, self.stats[['fold']],
    #                       left_on='id', right_index=True)



def preprocess(annotated_peptides,
               min_runs_no=5,
               _DataPreprocessor={},
               _get_stats={}):
    """Wrapper around preprocessing of the annotated peptides.""" 
    d = DataPreprocessor(annotated_peptides, **_DataPreprocessor)
    d.get_stats(min_runs_no=5, **_get_stats)
    d.get_distances_to_stats()
    return d
