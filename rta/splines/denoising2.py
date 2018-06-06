from multiprocessing import Pool, Process, Queue
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals


def denoise_and_align_run(annotated_run,
                          unlabelled_run,
                          formula,
                          model = 'Huber',
                          refit = True,
                          **kwds):
    """Remove noise and align the retention times in a run."""
    a, u = annotated_run, unlabelled_run

    # fit the spline
    model = spline(a, formula, **kwds)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides.
    if refit:
        model = spline(a[signal], formula)

    a_rt_aligned = np.array(a.rt) - predict(model, rt=a.rt) 
    u_rt_aligned = np.array(u.rt) - predict(model, rt=u.rt) 

    return signal, a_rt_aligned, u_rt_aligned



# suprisingly, this works!
def denoise_and_align(annotated, 
                      unlabelled,
                      formula,
                      model='Huber',
                      refit=True,
                      workers_cnt=16):
    """Denoise and align all runs."""

    def iter_groups():
        for run_no, a in annotated.groupby('run'):
            u = unlabelled[unlabelled.run == run_no]
            yield a, u, formula, model, refit

    with Pool(workers_cnt) as workers:
        res = workers.starmap(denoise_and_align_run, iter_groups())

    return res





def denoise_and_align_run2(run_cnt,
                           annotated,
                           unlabelled,
                           formula,
                           model = 'Huber',
                           refit = True):
    """Remove noise and align the retention times in a run."""
    a = annotated.loc[annotated.run == run_cnt,]
    u = unlabelled.loc[unlabelled.run == run_cnt,]
    
    # fit the spline
    model = spline(a, formula)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides.
    if refit:
        model = spline(a[signal], formula)

    # a_rt_aligned = np.array(a.rt) - predict(model, rt=a.rt) 
    # u_rt_aligned = np.array(u.rt) - predict(model, rt=u.rt) 

    # return signal, a_rt_aligned, u_rt_aligned
    return 0



def denoise_and_align_run_wrapper(tasks, returns):
    while True:
        task = tasks.get()
        if task == "end":
            return
        returns.put(denoise_and_align_run(*task))


def denoise_and_align2(annotated, 
                       unlabelled,
                       runs_no,
                       formula,
                       model='Huber',
                       refit=True,
                       workers_cnt=16):
    """Denoise and align all runs."""

    tasks = Queue()
    for run_no, a in annotated.groupby('run'):
        u = unlabelled[unlabelled.run == run_no]
        tasks.put((a, u, formula, model, refit))

    for _ in range(workers_cnt):
        tasks.put("end")

    returns = Queue()
    processes = [Process(target=denoise_and_align_run_wrapper,
                         args=(tasks, returns,)) 
                 for run_cnt in range(runs_no)]    

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    out = []
    while not returns.empty():
        out.append(returns.get())
    
    return out


