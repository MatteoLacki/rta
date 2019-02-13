%load_ext autoreload
%autoreload 2

import matplotlib.pyplot    as plt
import matplotlib
import numpy                as np
import os
import pandas               as pd
import seaborn              as sns

from rta.config             import *
from rta.isoquant           import retrieve_data, run_2_names_from
from rta.read_in_data       import big_data
from rta.quality_control.process_project    import process_retrieved_data
from rta.serialization.isoquant_data        import dump_retieved, load_retrieved


projects = {
    "Proj__15272392369260_8293106731954075_100_5":
    {
        'title': 'Nano 120',
        'runs' : ['S180424_02', 'S180424_03', 'S180424_04'],
        't_g'  : 93.0,
        't'    : 120
    },
    'Proj__15272392369260_8293106731954075_100_6':
    {
        'title': 'Nano 60',
        'runs' : ['S180424_06', 'S180424_07', 'S180424_08'],
        't_g'  : 36,
        't'    : 60
    },
    'Proj__15272392369260_8293106731954075_100_7':
    {
        'title': 'Nano 35 min',
        'runs' : ['S180424_09', 'S180424_10', 'S180424_11'],
        't_g'  : 18.5,
        't'    : 35
    },
    'Proj__15264893889320_6353458109334729_100_8':
    {
        'title': 'Micro 120',
        'runs' : ['S180427_20', 'S180427_21', 'S180427_22'],
        't_g'  : 106,
        't'    : 120
    },
    'Proj__15264893889320_6353458109334729_100_11':
    {
        'title': 'Micro 60 min',
        'runs' : ['S180427_05', 'S180427_06', 'S180427_07'],
        't_g'  : 50.5,
        't'    : 60
    },
    'Proj__15264893889320_6353458109334729_100_15':
    {
        'title': 'Micro 35',
        'runs' : ['S180427_30', 'S180427_31', 'S180427_32'],
        't_g'  : 28,
        't'    : 35
    },
    'Proj__15260213186990_6379462481554944_100_10':
    {
        'title': 'Micro 120 min No PCS',
        'runs' : ['S180507_10', 'S180507_11', 'S180507_12'],
        't_g'  : 106,
        't'    : 120
    },
    'Proj__15264893889320_6353458109334729_100_14':
    {
        'title': 'MF\n0.5μg',
        'runs' : ['S180427_27', 'S180427_28', 'S180427_29'],
        't_g'  : 28,
        't'    : 35
    }
}


try:
    folder = "/home/matteo/Projects/rta/data2"
    data = {p: load_retrieved(os.path.join(folder, p)) for p in projects}
except FileNotFoundError:
    data = {p: retrieve_data(password  = password,
                             user      = user,
                             ip        = ip,
                             project   = p,
                             verbose   = True, 
                             metadata  = True) for p in projects}
    for p in projects:
        dump_retieved(data[p],
                      f'/home/matteo/Projects/rta/data2/{p}')
