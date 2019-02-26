import pandas as pd
from pathlib import Path


def get_data():
	"""Wish a happy stroke to any functional purist."""
	data = Path("~/Projects/rta/data").expanduser()
	D = pd.read_msgpack(data/"D.msg")
	U = pd.read_msgpack(data/"U.msg")
	return D, U, D.run, D.mass, D.charge, D.rt, D.rta, D.dt, D.id