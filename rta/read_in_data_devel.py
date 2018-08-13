"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import  matplotlib.pyplot   as      plt

from rta.config             import *
from rta.plotters.runs      import plot_runs,\
                                   plot_experiment_comparison
from rta.quality_control.process_project import process_project

# # the first HELA dataset I've analysed.
# data_path = "../../../Data/annotated_and_unanottated_data.csv"

# # Ute's data-sets for microflow.
# data_path = "~/ms/Matteo/4Ute/2016-141 HYE Microflow_20180716_MF_120min_paper.csv"
# data = pd.read_csv(data_path)

# First shoot, than ask.
mass_projects = ["Proj__15272392369260_8293106731954075_100_1",
                 "Proj__15272392369260_8293106731954075_100_2",
                 "Proj__15272392369260_8293106731954075_100_3",
                 "Proj__15264893889320_6353458109334729_100_8",
                 "Proj__15264893889320_6353458109334729_100_11",
                 "Proj__15260213186990_6379462481554944_100_8",
                 "Proj__15260213186990_6379462481554944_100_10",
                 "Proj__15272392369260_8293106731954075_100_8",
                 "Proj__15264893889320_6353458109334729_100_17",
                 "Proj__15264893889320_6353458109334729_100_18"]

results = {}
for project in mass_projects:
    results[project] = process_project(project, password, user, ip)

plot_runs(c.D, title=title, run_2_name=run_2_name)
plot_experiment_comparison()

c, run_2_name, project, title = results[mass_projects[-1]]
plot_runs(c.D, title=title, run_2_name=run_2_name)


plot_experiment_comparison(results,
                           ["Proj__15272392369260_8293106731954075_100_8",
                            "Proj__15264893889320_6353458109334729_100_17"])

plot_experiment_comparison(results,
    ["Proj__15264893889320_6353458109334729_100_8", 
     "Proj__15260213186990_6379462481554944_100_10",
     "Proj__15272392369260_8293106731954075_100_1"],
     plt_style = 'default')
