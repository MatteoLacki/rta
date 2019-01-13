"""Testing if the populations of sequenced and unsequenced signals occupy roughly the same positions.
"""

%load_ext autoreload
%autoreload 2

from rta.read_in_data import big_data
from rta.preprocessing import preprocess, filter_unfoldable

annotated_all, unlabelled_all = big_data()

len(annotated_all)
len(unlabelled_all)





# Row data.
d = preprocess(annotated_all, min_runs_no)

folds_no = 10
min_runs_no = 5
d = filter_unfoldable(d, folds_no)