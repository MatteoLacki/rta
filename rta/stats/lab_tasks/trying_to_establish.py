"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import  matplotlib.pyplot   as      plt
from sqlalchemy import create_engine


from rta.config  import *
from rta.isoquant import retrieve_data
# # the first HELA dataset I've analysed.
# data_path = "../../../Data/annotated_and_unanottated_data.csv"

# # Ute's data-sets for microflow.
# data_path = "~/ms/Matteo/4Ute/2016-141 HYE Microflow_20180716_MF_120min_paper.csv"
# data = pd.read_csv(data_path)

# First shoot, than ask.
mass_projects = []
results = {}
ip = "192.168.1.196"
project = "Proj__13966189271230_9093339492956815"


data = retrieve_data(password,
                     project,
                     user,
                     ip,
                     verbose=True)

from sqlalchemy import create_engine

engine = create_engine(f"mysql+pymysql://{user}:{password}@{ip}:3306/{project}",
                           echo=True)
existing_table_names = engine.table_names()


engine = create_engine(f"mysql+pymysql://root:ykv16@{ip}",
                           echo=True)
existing_table_names = engine.table_names()


