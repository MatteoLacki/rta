"""Cross validate the input model."""

%load_ext autoreload
%autoreload 2
%load_ext line_profiler
from collections import Counter as count

from rta.models.base_model import coef, predict, fitted, coefficients, residuals, res, cv
from rta.models.plot import plot
from rta.models.SQSpline import SQSpline
from rta.read_in_data import data_folder, big_data
from rta.preprocessing import preprocess

# data = pd.read_csv(data_folder("one_run_5_folds.csv"))
# chunks_no = 20
# s_model = SQSpline()
# s_model.df_2_data(data, 'rt', 'rt_median_distance')
# s_model.fit(chunks_no=chunks_no)
annotated_all, unlabelled_all = big_data()
annotated_cv, annotated_cv_stats, runs_cnts = preprocess(annotated_all,
                                                         min_runs_no = 5)
count(annotated_cv.fold)

# at least this produces the outcome.
def get_folded_data(data, folds_no):
    AND = np.logical_and
    runs = np.unique(data.run)
    folds = np.arange(folds_no)
    for run in runs:
        for fold in folds:
            train = data.loc[AND(data.run == run, data.run != fold),:]
            test = data.loc[AND(data.run == run, data.run == fold),:]
            yield run, train, test

# 364 ms for this! This is long for such things.
x = list(get_folded_data(annotated_cv_slim, folds_no))

