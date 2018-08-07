"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

from collections import Counter
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

from rta.read_in_data import col_names,\
                             col_names_unlabelled
from rta.isoquant     import retrieve_data
from rta.config       import *

# # the first HELA dataset I've analysed.
# data_path = "../../../Data/annotated_and_unanottated_data.csv"

# # Ute's data-sets for microflow.
# data_path = "~/ms/Matteo/4Ute/2016-141 HYE Microflow_20180716_MF_120min_paper.csv"
# data = pd.read_csv(data_path)


project = "Proj__15272392369260_8293106731954075_100_1"
data    = retrieve_data(password  = password,
                        user      = user,
                        project   = project,
                        verbose   = True)

# two different ways to split a DataFrame.
# data that has not been given any sequence.
unlabelled = data.loc[data.sequence.isna(),
                      col_names_unlabelled].copy()
# all the identified peptides.
labelled   = data.dropna(subset = ['sequence']).copy()
labelled['id'] = labelled['sequence'] + " " + labelled['modification'].astype(str)

# filter out peptides that have more than one instance while conditioning on id, run and type.
id_run_type_cnt = labelled.groupby(['id', 'run', 'type']).id.count()


len(id_run_type_cnt[id_run_type_cnt > 1])

labelled.merge(
    pd.DataFrame(id_run_type_cnt[id_run_type_cnt == 1]),
    on = ('id', 'run', 'type'))


pd.DataFrame(id_run_type_cnt[id_run_type_cnt == 1]).index

pd.DataFrame(id_run_type_cnt[id_run_type_cnt == 1]).join(

    )
