import numpy as np
import pandas as pd
from pathlib import Path

from rta.reference import cond_medians
from rta.filters.angry import is_angry


def get_data():
    """Get data after filtering, ready for clustering."""
    data = Path("~/Projects/rta/data").expanduser()
    D = pd.read_msgpack(data/"D.msg")
    U = pd.read_msgpack(data/"U.msg")
    mass_me = cond_medians(D.mass, D.id)
    rt_me = cond_medians(D.rt, D.id)
    dt_me = cond_medians(D.dt, D.id)
    D_mass = mass_me - D.mass
    D_rta = rt_me - D.rta
    D_dt = dt_me - D.dt
    angry_mass = is_angry(D_mass)
    angry_rta = is_angry(D_rta)
    angry_mass_rta = np.logical_or(angry_mass, angry_rta)
    U = U.append(D.loc[angry_mass_rta, U.columns])
    D = D.loc[~angry_mass_rta,]
    return D, U, D.run.values, D.mass.values, D.charge.values,\
        D.rt.values, D.rta.values, D.dt.values, D.id.values