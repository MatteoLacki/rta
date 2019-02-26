import pandas as pd
from pathlib import Path

from rta.read.csvs import big_data
from rta.preprocessing import preprocess
from rta.dev_get_rta import data_for_clustering

data = Path("~/Projects/rta/data/").expanduser()

annotated_all.to_msgpack(data/"annotated_all.msg")
unlabelled_all.to_msgpack(data/'unlabelled_all.msg')
D, U = data_for_clustering(annotated_all, 5, unlabelled_all)
D.to_msgpack(data/"D.msg")
U.to_msgpack(data/"U.msg")
