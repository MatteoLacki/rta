import numpy as np

from rta.models.splines.robust import RobustSpline


def trial_fit(chunks_no=2):
    m = RobustSpline()
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y = np.array([1.1, 1.2, 1.15, 1.54, 1.7, 1.543, 1.67, 1.64, 1.725,  1.6454,  2.11,  1.98,  2.12])
    folds = np.array([0,1,0,1,0,1, 0,1,0,1,0,1, 0])
    m.fit(x, y, chunks_no=chunks_no)
    m.cv(folds)
    return m


def test_RobustSpline():
    """Check if the algorithm replicates the old results."""
    m = trial_fit(chunks_no=2)
    signals = np.array([False,True,False,True,True,True,True,True,True,True,False,True,False])
    cv_stats = np.array([[0.211267, 0.043517],
                         [0.211267, 0.043517],
                         [0.034872, 0.005300]])
    assert np.sum(np.abs(cv_stats - m.cv_stats)).sum() < 1e-5, "Fold statistics significantly differ."
