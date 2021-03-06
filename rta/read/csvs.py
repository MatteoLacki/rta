# import csv
#
# with open('data/pure_data.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in reader:
#         print(', '.join(row))
import numpy as np
import pandas as pd
import platform
import os

from rta.data.column_names import vars_annotated, vars_unlabelled

# TODO eliminate the 'id' column and use indexing based on 
# sequence and modification instead (to save memory)


def data_folder(path=""):    
    system = platform.system()
    if system == "Linux":
        pre_path = "rta/data/"
    elif system == "Darwin":
        pre_path = "~/Projects/retentiontimealignment/Data/"
    else:
        raise KeyError("We support MacOS and Linux only.")
    path = os.path.join(pre_path, path)
    return path    


def big_data(path = "",
             vars_annotated  = vars_annotated,
             vars_unlabelled = vars_unlabelled):
    """Read in the big data for the project.

    TODO: this might be turned into some example set.
    """
    path = data_folder(path)
    annotated  = pd.read_csv(path+'annotated_data.csv',
                             usecols=vars_annotated)
    unlabelled = pd.read_csv(path+'unannotated_data.csv',
                             usecols=vars_unlabelled)
    return annotated, unlabelled

# LearnPair = namedtuple('LearnPair', 'training test')
# def x_validate(X, folds=10, id = ['id']):
#     gkf = GroupKFold(n_splits=folds)
#     for train, test in gkf.split(X, groups=X[id]):
#         yield LearnPair(X.iloc[train], X.iloc[test])

# path = "~/Projects/retentiontimealignment/Data/"
# variables = {'run': np.int8, 
#   'mass': np.float64, 
#   'intensity': np.float64, 
#   'charge': np.int8,
#   'FWHM': np.float64, 
#   'rt': np.float64,
#   'dt': np.float64,
#   'LiftOffRT': np.float64, 
#   'InfUpRT': np.float64,
#   'TouchDownRT': np.float64,
#   'sequence': 'U', 
#   'modification': 'U', 
#   'type': 'U', 
#   'score': np.float64}


col_names = ['run',
             'mass',
             'intensity',
             'charge',
             'FWHM',
             'rt',
             'dt',
             'LiftOffRT',
             'InfUpRT',
             'InfDownRT',
             'TouchDownRT',
             'sequence',
             'modification',
             'type',
             'score']


col_names_unlabelled = [ 'run',
                         'mass',
                         'intensity',
                         'charge',
                         'FWHM',
                         'rt',
                         'dt',
                         'LiftOffRT',
                         'InfUpRT',
                         'InfDownRT',
                         'TouchDownRT']

