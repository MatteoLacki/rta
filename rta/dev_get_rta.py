from rta.preprocessing import preprocess
from rta.reference import choose_statistical_run
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedian

#TODO: DF should not have any of these silly indices?
# But this will not be efficient.
def align_rt(annotated_all, min_runs_per_id, unlabelled_all):
    """Get data with aligned retention times."""
    D, stats, pddra, pepts_per_run = preprocess(annotated_all, min_runs_per_id) # this will be done in MariaDB/Parquet
    X, uX = choose_statistical_run(D, 'rt', 'median')
    x, y, g = X.x.values, X.y.values, X.run.values
    runs = D.run.unique()
    m = {r: RollingMedian() for r in runs} # each run can have its own model
    M = BigModel(m)
    M.fit(x, y, g) # this should work also if run is in the index???
    X['rta'] = M(x, g)
    X = X.reset_index().set_index(['id', 'run'])
    D = D.reset_index().set_index(['id', 'run'])
    D = D.join(X.rta)
    # this looks really akward...
    # maybe better to have a function accepting tuples?
    # always more natural.
    D.reset_index(inplace=True)
    D.set_index('id', inplace=True)
    U = unlabelled_all[['run', 'mass', 'intensity', 'rt', 'dt']].copy()
    U['rta'] = M(U.rt, U.run)
    return D, U

