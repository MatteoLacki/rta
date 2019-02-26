import numpy as np

from rta.preprocessing import preprocess
from rta.reference import cond_medians
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedian


def data_for_clustering(annotated_all, min_runs_per_id, unlabelled_all):
    """ """
    D, stats, pddra, pepts_per_run = preprocess(annotated_all, min_runs_per_id)
    rt = D.rt.values
    ids = D.id.values
    g = D.run.values
    rt_me, _ = cond_medians(rt, ids)
    D['rt_me'] = rt_me
    runs = np.unique(g)
    B = BigModel({r: RollingMedian() for r in runs})
    B.fit(rt, rt_me, g)
    D['rta'] = B(D.rt, D.run)
    U = unlabelled_all[['run', 'mass', 'intensity', 'rt', 'dt']].copy()
    U['rta'] = B(U.rt, U.run)
    return D, U

