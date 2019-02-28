%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from pathlib import Path

from rta.reference import cond_medians
from rta.filters.angry import is_angry
from rta.models.big import BigModel
from rta.models.rolling_median import RollingMedianSpline
from rta.plot.runs import plot_distances_to_reference


data = Path("~/Projects/rta/data").expanduser()
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

mass = D.mass.values
rt = D.rt.values
dt = D.dt.values
ids = D.id.values
run = D.run.values
runs = np.unique(run)

mass_me = cond_medians(mass, ids)
rt_me = cond_medians(rt, ids)
dt_me = cond_medians(dt, ids)

B = BigModel({r:RollingMedianSpline() for r in runs})
rt0 = rt
rt_me0 = rt_me
B.fit(rt0, rt_me0, run)
# B.plot(s=1, residuals=True)

# this gives the alignment.
# maybe change res to ... these are correct!!!
# B(rt0, run) == rt0 + B.res()
# B(rt0, run) == B.fitted()
rt1 = B.fitted()
rt_me1 = cond_medians(rt1, ids)

plot_distances_to_reference(rt1, rt_me1, run, s=1)
plot_distances_to_reference(rt0, rt_me1, run, s=1, add_line=False)
plot_distances_to_reference(rt0, rt_me0, run, s=1, add_line=False)

C = BigModel({r:RollingMedianSpline() for r in runs})
C.fit(rt0, rt_me1, run)
rt1_alt = C.fitted()
C.plot(s=1)
C.plot(s=1, residuals=True)

