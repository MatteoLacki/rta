# import csv
#
# with open('data/pure_data.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in reader:
#         print(', '.join(row))
import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from collections import namedtuple

DT = pandas.read_csv('rta/data/pure_data.csv')


LearnPair = namedtuple('LearnPair', 'training test')

def x_validate(X, folds=10, id = ['id']):
    gkf = GroupKFold(n_splits=folds)
    for train, test in gkf.split(X, groups=X[id]):
        yield LearnPair(X.iloc[train], X.iloc[test])

# Y = list(x_validate(DT))
# len(Y)
# Y[0].test
