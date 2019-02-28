import numpy as np
import pandas as pd
from pathlib import Path

from rta.reference import cond_medians
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedian, RollingMedianSpline
from rta.plot.runs import plot_distances_to_reference

data = Path("~/Projects/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

rt = D.rt.values
ids = D.id.values
rt_me = cond_medians(rt, ids)
run = D.run.values
runs = np.unique(run)

B = BigModel({r: RollingMedianSpline() for r in runs})
B.fit(rt, rt_me, run)
# B.res() == B(rt, run) - rt
# rt + B.res() == B.fitted()
plot_distances_to_reference(rt, B.fitted(), run)


