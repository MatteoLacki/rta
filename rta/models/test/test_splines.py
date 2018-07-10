%load_ext autoreload
%autoreload 2
%load_ext line_profiler
import numpy as np

from rta.models.splines.robust import RobustSpline, mad_window_filter
from rta.models import plot

model = RobustSpline()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y = np.array([1.1, 1.2, 1.15, 1.54, 1.7, 1.543, 1.67, 1.64, 1.725,  1.6454,  2.11,  1.98,  2.12])

model.fit(x, y, chunks_no=chunks_no)
model.signal

# plot(model)
folds = np.array([0,1,0,1,0,1, 0,1,0,1,0,1, 0])
model.cv(folds)



def test_RobustSpline():
    """Check if the algorithm replicates the old results."""
    pass


from rta.array_operations.misc import overlapped_percentile_pairs
from rta.array_operations.misc import percentiles_of_N_integers

list(overlapped_percentile_pairs(len(x), 1))
list(percentiles_of_N_integers(len(x), 1))




