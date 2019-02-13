"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import  matplotlib.pyplot   as      plt
import  numpy               as      np

from rta.config             import *
from rta.plotters.runs      import plot_runs,\
                                   plot_experiment_comparison,\
                                   plot_runs_individually
from rta.quality_control.process_project import process_project


mass_project = "Proj__15264893889320_6353458109334729_100_18"
result       = process_project(project, password, user, ip)

nano_project = "Proj__15272392369260_8293106731954075_100_8"
nano_result  = process_project(nano_project, password, user, ip)

projects = ["Proj__15264893889320_6353458109334729_100_18",
            "Proj__15272392369260_8293106731954075_100_8"]



c, run_2_name, project, title = result
plot_runs(c.D,
          title      = title,
          run_2_name = run_2_name)


c, run_2_name, project, title = nano_result
plot_runs(c.D,
          title      = title,
          run_2_name = run_2_name)


results = {}
results[mass_project] = result
results[nano_project] = nano_result

plot_experiment_comparison(results,
                           [mass_project, nano_project])

runs = result[0].D.run.unique()
D    = result[0].D
run_2_name = result[1]

plot_runs_individually(D, runs, 5, 3, run_2_name)

# another thing: select runs for analysis:




# investigating the per-run percentiles.
# c.D.groupby('run').rt_0.apply(np.percentile, q=[2.5, 97.5])
# c.D.groupby('run').rt_0.agg({'perc2.5': lambda x: np.percentile(x, 2.5),
#                              'perc97.5': lambda x: np.percentile(x, 97.5)})

