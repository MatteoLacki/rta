%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.config             import *
from rta.isoquant           import retrieve_data, run_2_names_from
from rta.read_in_data       import big_data
from rta.quality_control.process_project import process_retrieved_data
from rta.serialization.isoquant_data import dump_retieved, load_retrieved

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
    }
}




retrieved_data = [retrieve_data(password  = password,
                                user      = user,
                                ip        = ip,
                                project   = p,
                                verbose   = True, 
                                metadata  = True) for p in projects]

for rd, p in zip(retrieved_data, projects):
    projects[p]['data'] = rd

# additional data:
p_add  = "Proj__15260213186990_6379462481554944_100_10"
rd_add = retrieve_data(password  = password,
                       user      = user,
                       ip        = ip,
                       project   = p_add,
                       verbose   = True, 
                       metadata  = True)
projects[p_add] = {'title' : 'Micro 120 min No PCS',
                   'runs'  : ['S180507_10', 'S180507_11', 'S180507_12'],
                   't_g'   : 106,
                   't'     : 120,
                   'data'  : rd_add}


# dump all intermediate results.

experiments = ''
for p in projects:
    dump_retieved(projects[p]['data'],
                  f'/home/matteo/Projects/retentiontimealignment/Data/{experiments}/{p}')


# data, proj_rep, worklow_rep = projects[p_add]['data']
# dump_retieved(projects[p_add]['data'],
#               '/home/matteo/Projects/retentiontimealignment/Data/test')
# folder = '/home/matteo/Projects/retentiontimealignment/Data/test/'

# load_retrieved(folder)
# dump results to something.
# testing pyarrow: don't know if it's worth it...
# import sys
# table = pa.Table.from_pandas(data)
# data.to_csv('/home/matteo/Projects/retentiontimealignment/Data/test.csv')
# pq.write_table(table, '/home/matteo/Projects/retentiontimealignment/Data/test.parquet')
# pq.write_table(table,
#                '/home/matteo/Projects/retentiontimealignment/Data/test.gzip',
#                compression='gzip')
# pq.write_table(table,
#                '/home/matteo/Projects/retentiontimealignment/Data/test.brotli',
#                compression='brotli')

import seaborn as sns


to_plot = projects
to_plot = ['Proj__15272392369260_8293106731954075_100_6', 'Proj__15264893889320_6353458109334729_100_15']
f, axes = plt.subplots(1, len(to_plot),
                       figsize=(7, 7),
                       sharex=True, sharey=True)

for i, p in enumerate(to_plot):
    data, proj_rep, worklow_rep = projects[p]['data']
    ax = sns.boxplot(x    = "run", 
                     y    = "FWHM",
                     data = data,
                     ax   = axes[i])
    ax.set_title(projects[p]['title'])
plt.show()




# run2name, name2run = run_2_names_from(retrieved_data[0][2])
# selecting the proper runs: no need - there are only those needed anyway.
# retrieved_data[0][2]
# retrieved_data[1][2]
# retrieved_data[2][2]
# retrieved_data[3][2]
# retrieved_data[4][2]
# retrieved_data[5][2]

# data, proj_rep, workflow_rep = projects["Proj__15272392369260_8293106731954075_100_5"]['data']

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
# ax1.hist(L[L.run == 1].FWHM, bins=100, density=True)
# ax2.hist(U[U.run == 1].FWHM, bins=100, density=True)
# plt.show()

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
# L[L.run == 1].head()
# L1 = L[L.run == 1]
# U1 = U[U.run == 1]

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
# plt.suptitle('TouchDownRT - LiftOffRT')
# ax1.hist(L1.TouchDownRT - L1.LiftOffRT, bins=100, density=True)
# plt.title('Annotated')
# ax2.hist(U1.TouchDownRT - U1.LiftOffRT, bins=100, density=True)
# plt.title('Unlabelled')
# plt.show()




for p in projects:
    projects[p]['processed_data'] = process_retrieved_data(*projects[p]['data'],
                                                           project     = p,
                                                           min_runs_no = 3)

# calculating peak capacities
def get_peak_capacity_50_perc(FWHM, gradient_time):
    return 1 + gradient_time / (1.679 * np.mean(FWHM))

def get_peak_capacity_4_sigma(RT_SD, gradient_time):
    return 1 + gradient_time / (4.0 * np.mean(RT_SD))

def get_peak_capacity_diff_points(left_ends, right_ends, gradient_time):
    return 1 + gradient_time / np.mean(right_ends - left_ends)


# def get_peak_capacities():
c, run_2_name, project, title = projects['Proj__15272392369260_8293106731954075_100_5']['processed_data']
gradient_time = projects['Proj__15272392369260_8293106731954075_100_5']['t_g']
D = projects['Proj__15272392369260_8293106731954075_100_5']['data'][0]

D1 = D[D.run == 1]
D1.head()
get_peak_capacity_50_perc(D1.FWHM, gradient_time)


peak_caps = 1.0 + gradient_time / (1.679 * D1.FWHM)

s, e = np.percentile(D1.rt, q=[2.5, 97.5])
important_rts = (D1.rt > s) & (D1.rt < e)
plt.scatter(D1.rt[important_rts], peak_caps[important_rts], s = .1)
plt.show()

peak_caps.mean()

plt.scatter(D1.rt, peak_caps, s = .1)
plt.show()


# p = 'Proj__15272392369260_8293106731954075_100_5'
peak_capacities = {}
for p in projects:
    D = projects[p]['data'][0]
    run2name = projects[p]['processed_data'][1]
    runs = np.unique(D.run)
    gradient_time = projects[p]['t_g']
    for r in runs:
        d = D[D.run == r]
        peak_capacities[(p, run2name[r])] = get_peak_capacity_50_perc(d.FWHM, gradient_time)


Nano  = 'Proj__15272392369260_8293106731954075_100_7'
Micro = 'Proj__15264893889320_6353458109334729_100_15'

NanoD = projects[Nano]['data'][0]
NanoD1 = NanoD[NanoD.run == 1]

MicroD = projects[Micro]['data'][0]
MicroD1 = MicroD[MicroD.run == 1]


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.hist(NanoD1.FWHM, bins=100, density=True)
ax2.hist(MicroD1.FWHM, bins=100, density=True)
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax1.hist(NanoD1.TouchDownRT - NanoD1.LiftOffRT, bins=100, density=True)
ax2.hist(MicroD1.TouchDownRT - MicroD1.LiftOffRT, bins=100, density=True)
plt.show()




for run, d in annotated_all.groupby('run'):
    get_peak_capacity_50_perc(d.FWHM,  gradient_time)
    get_peak_capacity_4_sigma(d.rt_sd, gradient_time)
    get_peak_capacity_diff_points(d.)