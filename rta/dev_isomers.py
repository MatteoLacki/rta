"""Checking if the pure indices cannot outcompete the silly kd-tree."""
%load_ext autoreload
%autoreload 2

from math import inf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from plotnine import *
from collections import Counter
import re

from rta.array_operations.dataframe_ops import get_hyperboxes, conditional_medians, normalize
from rta.array_operations.non_overlapping_intervals import OpenOpen, OpenClosed, get_intervals_np
from aa2atom.aa2atom import aa2atom, atom2str
from aa2atom.ptms import PLGSptms2atom, ptms

data = Path("~/Projects/rta/rta/data").expanduser()
A = pd.read_msgpack(data/"A.msg")
D = pd.read_msgpack(data/"D.msg")
U = pd.read_msgpack(data/"U.msg")

SM = A.id.str.split(' ', n=1, expand=True)
SM.columns = ['sequence', 'modification']
SM['charge'] = A.charge

sm = SM.query('modification != "NA"').copy()

def aamod2atom(seq, mod):
	aa = aa2atom(seq)
	if mod is not 'NA':
		aa.update(PLGSptms2atom(mod))
	return atom2str(aa)

np_aamod2atom = np.vectorize(aamod2atom)
sm['formula'] = np_aamod2atom(sm.sequence, sm.modification)

x = pd.DataFrame(sm.groupby(['formula', 'charge']).size())
x.columns = ['peps_cnt']
x = x.reset_index()

y = x.groupby(['charge','peps_cnt']).size()
y = pd.DataFrame(y)
y.columns = ['cnt']
y = y.reset_index()


( ggplot(y, aes('peps_cnt', 'cnt')) + 
	geom_bar(stat='identity') + 
	facet_grid('.~charge', scales='free_y') +
	xlab('Isomeric Peptides per Formula') +
	ylab('Number of different formulas'))

( ggplot(y.query('peps_cnt > 1'), aes('peps_cnt', 'cnt')) + 
	geom_bar(stat='identity') + 
	facet_grid('.~charge', scales='free_y') +
	xlab('Isomeric Peptides per Formula') +
	ylab('Number of different formulas'))


y['isomeric'] = np.where(y.peps_cnt > 1, 'isomeric', 'non-isomeric')
w = y.groupby(['charge', 'isomeric']).cnt.sum()
w = pd.DataFrame(w)
w = w.reset_index()
w.pivot(index='charge',columns='isomeric',values='cnt')