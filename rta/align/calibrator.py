"""A class for calibrating the parameters of the model. """
from rta.models.base_model import fitted

from rta.cv.folds import stratified_group_folds
from rta.cv.folds import replacement_folds_strata

class Calibrator(object):
    def __init__(self, 
                 preprocessed_data,
                 feature,
                 run='run'):
        """Initialize the Calibrator.

        Args:
            preprocessed_data (pandas.DataFrame): data to assign folds to.
            feature (string):   the name of the feature in the column space of the preprocessed_data that will be aligned.
        """
        self.dp = preprocessed_data
        self.feature = feature
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

        # filter out peptides that occur in runs in groups smaller than ''
        self.dp.filter_unfoldable_strata(folds_no)
        self.folds_no = folds_no # write a setter to check for proper type and value

        # TODO: this should really be some wrapper around the silly method.
        if fold.__name__ == 'stratified_group_folds':
            # we want the result to be sorted w.r.t. median rt.
            self.dp.stats.sort_values(["runs",
                                       self.dp.stat_name+'_'+feature],
                                      inplace=True)

        # get fold assignments for pept-id groups
        self.dp.stats['fold'] = fold(self.dp.strata_cnts,
                                     self.folds_no,
                                     shuffle)

        # if there was some other fold in the dataframe - drop it!
        self.dp.D.drop(labels = [c for c in self.dp.D.columns
                                 if 'fold' in c], 
                       axis = 1,
                       inplace = True)

        # propage fold assignments to the main data
        dp.D = pd.merge(dp.D, dp.stats[['fold']],
                        left_on='id', right_index=True)

    def cv_run_param(self):
        pass

    def iter_run_param(self):
        """Iterate over the data runs and fitting parameters."""
        for r, d_r in self.dp.D.groupby(sef.run):
            d_r = d_r.sort_values(feature)
            d_r = d_r.drop_duplicates(feature)
            for p in parameters:
                out = [r, d_r, p, self.dp.folds]
                out.extend(_cv_run_args)
                yield out

    def calibrate(self,
                  parameters=None,
                  cores_no=cpu_count()):
        if not parameters:
            parameters = [{"chunks_no": 2**e} for e in range(2,8)]
        self.parameters = parameters

        with Pool(cores_no) as p:
            results = p.starmap(self.cv_run_param,
                                self.iter_run_param())

        best_model = select_best_model(results)
    # align the given dimension
        self.dp['aligned_' + self.feature] = fitted(best_model)
        return self.dp, results



def calibrate(preprocessed_data,
              feature='rt',
              folds_no=10,
              fold=stratified_group_folds,
              shuffle=True):
    """Calibrate the given feature of the data."""
    calibrator = Calibrator(preprocessed_data, feature)
    calibrator.set_folds(folds_no, fold, shuffle)
    calibrator.calibrate()
    return calibrator

