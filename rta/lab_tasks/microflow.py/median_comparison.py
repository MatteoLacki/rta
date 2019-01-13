"""Plot differences to medians."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy             as np
import pandas as pd
from os.path import join as pjoin

from rta.align.calibrator       import calibrate
from rta.preprocessing          import preprocess, filter_unfoldable
from rta.isoquant               import run_2_names_from
from rta.plotters.runs          import plot_runs



def save_results(runs, filename, annotated_all, run_2_name):
    annotated_all = annotated_all[np.isin(annotated_all.run, runs)].copy()
    folds_no    = 10
    min_runs_no = 5
    d = preprocess(annotated_all, min_runs_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrate(feature     = 'rt',
                  data        = d,
                  folds_no    = folds_no,
                  min_runs_no = min_runs_no,
                  align_cnt   = 0)
    c.D['names'] = [run_2_name[r] for r in c.D.run]
    c.D.to_csv(
        pjoin("/Users/matteo/Projects/nano_vs_micro/data/results/", proj, filename),
        index=False)



if __name__ == "__main__":
    path = "/Users/matteo/Projects/nano_vs_micro/data/projects/"

    # micro runs
    proj = "Proj__15264893889320_6353458109334729_100_18"
    # creating the peptide_id, which is for now required.
    D = pd.read_csv(pjoin(path, proj, 'all_signals.csv'))
    annotated_all = D[D.sequence.notnull()].copy()
    annotated_all['id'] = annotated_all.sequence + "_" + annotated_all.modification.astype(str) + "_" + annotated_all.TYPE

    workflow_report = pd.read_csv(pjoin(path, proj, 'workflow_report.csv'))
    run_2_name, name_2_run = run_2_names_from(workflow_report)

    runs = [f"S180502_{i}" for i in range(31, 37)]
    runs = [name_2_run[n] for n in runs]
    save_results(runs, "S180502_31:36.csv", annotated_all, run_2_name)

    runs = [f"S180427_{i}" for i in range(20, 26)]
    runs = [name_2_run[n] for n in runs]
    save_results(runs, "S180427_20:25.csv", annotated_all, run_2_name)

    # nano runs
    proj = "Proj__15272392369260_8293106731954075_100_8"
    # creating the peptide_id, which is for now required.
    D = pd.read_csv(pjoin(path, proj, 'all_signals.csv'))
    annotated_all = D[D.sequence.notnull()].copy()
    annotated_all.rename({'type':'TYPE'}, axis='columns', inplace=True)
    annotated_all['id'] = annotated_all.sequence + "_" + annotated_all.modification.astype(str) + "_" + annotated_all.TYPE
    workflow_report = pd.read_csv(pjoin(path, proj, 'workflow_report.csv'))
    run_2_name, name_2_run = run_2_names_from(workflow_report)
    runs = [f"S180108_{i}" for i in range(30, 36)]
    runs = [name_2_run[n] for n in runs]
    save_results(runs, "S180108_30:35.csv", annotated_all, run_2_name)