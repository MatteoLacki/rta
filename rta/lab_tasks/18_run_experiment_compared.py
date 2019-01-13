"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import  matplotlib.pyplot   as      plt
import  numpy               as      np

from rta.config             import *
from rta.isoquant           import retrieve_data
from rta.plotters.runs      import plot_runs,\
                                   plot_experiment_comparison,\
                                   plot_runs_individually
from rta.quality_control.process_project import process_retrieved_data


projects = ["Proj__15264893889320_6353458109334729_100_18",
            "Proj__15272392369260_8293106731954075_100_8"]

retrieved_data = [retrieve_data(password  = password,
                                user      = user,
                                ip        = ip,
                                project   = p,
                                verbose   = True, 
                                metadata  = True) for p in projects]

# split the 
data0, proj_rep0, worklow_rep0 = retrieved_data[0]
run_2_name0 = dict(zip(worklow_rep0.workflow_index.values,
                      worklow_rep0.acquired_name.values))
S180502 = data0.loc[np.isin(data0.run, range(1,7)),  ].copy()
S180427 = data0.loc[np.isin(data0.run, range(10,16)),].copy()

data1, proj_rep1, worklow_rep1 = retrieved_data[1]
run_2_name1 = dict(zip(worklow_rep1.workflow_index.values,
                      worklow_rep1.acquired_name.values))
S180108 = data1.loc[np.isin(data1.run, range(1,7)),].copy()

names = ['Micro S180502', 'Micro S180427', 'Nano S180108']
final_data_1oo6 = [(S180502, proj_rep0, worklow_rep0, 'Micro S180502', 1),
                   (S180427, proj_rep0, worklow_rep0, 'Micro S180427', 1),
                   (S180108, proj_rep1, worklow_rep1, 'Nano S180108' , 1)]
final_data_6oo6 = [(S180502, proj_rep0, worklow_rep0, 'Micro S180502', 6),
                   (S180427, proj_rep0, worklow_rep0, 'Micro S180427', 6),
                   (S180108, proj_rep1, worklow_rep1, 'Nano S180108' , 6)]

processed_data_1oo6 = [process_retrieved_data(*rd) for rd in final_data_1oo6]
processed_data_6oo6 = [process_retrieved_data(*rd) for rd in final_data_6oo6]

results_1oo6 = dict(zip(names, processed_data_1oo6))
results_6oo6 = dict(zip(names, processed_data_1oo6))

c, run_2_name, project, title = results_6oo6['Micro S180502']
plot_runs(c.D,
          title      = title,
          run_2_name = run_2_name)

plot_experiment_comparison(results_1oo6, names)
plot_experiment_comparison(results_6oo6, names)




# investigating the per-run percentiles.
c.D.groupby('run').rt_0.apply(np.percentile, q=[2.5, 97.5])
c.D.groupby('run').rt_0.agg({'perc2.5': lambda x: np.percentile(x, 2.5),
                             'perc97.5': lambda x: np.percentile(x, 97.5)})
# Write a method to save the outcomes and plot them again.
c.D.head()
c.D.columns

