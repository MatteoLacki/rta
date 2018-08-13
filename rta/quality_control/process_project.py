from sqlalchemy             import create_engine

from rta.align.calibrator   import calibrate
from rta.isoquant           import retrieve_data
from rta.preprocessing      import filter_peptides_with_unique_types,\
                                   preprocess,\
                                   filter_unfoldable


def process_retrieved_data(data,
                           proj_rep,
                           worklow_rep,
                           project,
                           min_runs_no = 1):
    run_2_name = dict(zip(worklow_rep.workflow_index.values,
                      worklow_rep.acquired_name.values))
    title = proj_rep.title[0]
    L, U = filter_peptides_with_unique_types(data, False)
    # Adjust L to the old reasoning: we need a column id for coding convenience.
    L.reset_index(inplace = True)
    L['id'] = L['sequence'] + " " + L['modification'].astype(str) # old way
    used_runs = L.run.unique()
    folds_no  = len(used_runs)
    d = preprocess(L, folds_no)
    d = filter_unfoldable(d, folds_no)
    c = calibrate(feature     = 'rt',
                  data        = d,
                  folds_no    = folds_no,
                  min_runs_no = min_runs_no, 
                  align_cnt   = 1)
    return c, run_2_name, project, title


def process_project(project,
                    password,
                    user,
                    ip,
                    min_runs_no,
                    verbose=True):
    data, proj_rep, worklow_rep = retrieve_data(password  = password,
                                                user      = user,
                                                ip        = ip,
                                                project   = project,
                                                verbose   = verbose, 
                                                metadata  = True)
    return process_retrieved_data(data, proj_rep, worklow_rep, project, min_runs_no)