"""Investigation into the splitting of the peaks."""
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
from plotnine import *


unlabelled_all.groupby("run").mass.count().values

unlabelled_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/unlabelled_all.msg')
annotated_all = pd.read_msgpack('/Users/matteo/Projects/rta/data/annotated_all.msg')
runs = D.run.unique()

# for every point from annotated_all find all the closest points.
unlabelled_all.sort_values(['run', 'rt'], inplace=True)

Z = pd.concat([annotated_all[['run', 'rt', 'FWHM', 'LiftOffRT', 'TouchDownRT']],
               unlabelled_all[['run', 'rt', 'FWHM', 'LiftOffRT', 'TouchDownRT']]])

u10 = unlabelled_all.groupby('run').get_group(10)
a10 = annotated_all.groupby('run').get_group(10)
annotated_all['run_c'] = annotated_all['run'].astype('category')

(ggplot(a10, aes('dt', 'FWHM')) +
    geom_point(size=.5, alpha=.5) +
    theme_bw())

# small modes o
(ggplot(annotated_all, aes('FWHM', group='run_c', color='run_c')) +
    geom_density() +
    theme_bw())
(ggplot(annotated_all, aes('np.log(TouchDownRT - LiftOffRT)', group='run_c', color='run_c')) +
    geom_density() +
    theme_bw())

# OK, how does the presence of other close points in rt play with FWHM and RT?
Z1 = Z.groupby('run').get_group(1).reset_index(drop=True)
Z1.sort_values('rt', inplace=True)

# finding the closest rts to other rts.
z1rt = Z1.rt.values
Dz1rt = np.diff(z1rt)
closest_rt = np.zeros(z1rt.shape)
closest_rt[[0,-1]] = Dz1rt[[0,-1]]
closest_rt[1:-1] = np.minimum(Dz1rt[:-1], Dz1rt[1:])
Z1['d_min_rt'] = closest_rt

# this is a lot of points..
(ggplot(Z1, aes('FWHM', 'np.log(d_min_rt)')) + geom_point(size=.2, alpha=.1))

# Why are there patterns in the minimal distances???

(ggplot(Z1, aes('np.log(TouchDownRT - LiftOffRT)', 'np.log(d_min_rt)')) + geom_point(size=.2, alpha=.1))
