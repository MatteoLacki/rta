"""
Here we check, what is reported as mass.
Is it the monoisotopic mass?
"""
%load_ext autoreload
%autoreload 2

from math import inf; import numpy as np; import pandas as pd
from pathlib import Path; from collections import Counter
import matplotlib.pyplot as plt; from plotnine import *

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")

A_stat = pd.DataFrame(A.groupby(['id']).size()).reset_index()
A_stat.columns = ['id', 'runs']
A_stat_10 = A_stat.query('runs == 10')
A_id = A.set_index('id')
A_10 = A_id.loc[A_stat_10.id,:]
A_10.reset_index(inplace=True)

(ggplot(A_10.query("(mass > 1000) & (mass < 1100)")) +
	geom_point(aes(x='massa',
				y='dta',
				color='charge'),
				size=.2))

# "AAAAAAAAAPAAAATAPTTAATTAATAAQ" = C99H166N30O37
most_probable = 2368.206
monoisotopic = 2367.203


A_10.id[21]
A_10[A_10.id == "AAAAAAAAAPAAAATAPTTAATTAATAAQ NA"].mass.values
A_10[A_10.id == 'AAAAAAALQAK NA']
A_10[A_10.id == 'AAADALSDLEIK NA'].mass.values

A_10sorted = A_10.sort_values(by='mass')
A_10sorted[A_10sorted.modification.isna()]
A_10[A_10.id == 'IPVTDEEQTNVPYIYAIGDILEDKVELTPVAIQAGR NA'].mass.values



(ggplot(A_10.query("(mass > 1000) & (mass < 1100)")) +
	geom_point(aes(x='rta',
				y='dta',
				color='charge',
				label='id'),
				size=1))

# Look into the R...
# It seems that what is done is to take the mass of the top peak, divide it.