from multiprocessing import Pool
import numpy as np
from sklearn import mixture

from rta.models import spline
from rta.models import predict, fitted, coef, residuals


def denoise_and_align_run(annotated_run,
                          unlabelled_run,
                          formula,
                          model = 'Huber',
                          refit = True,
                          return_model = False):
    """Remove noise and align the retention times in a run."""
    a, u = annotated_run, unlabelled_run

    # fit the spline
    model = spline(a, formula)

    # fit the Gaussian mixture
    res = residuals(model).reshape((-1,1))
    gmm = mixture.GaussianMixture(n_components=2, # only 2: noise & signal
                                  covariance_type='full').fit(res)

    # signal has smaller variance
    signal_idx, noise_idx = np.argsort(gmm.covariances_.ravel())
    signal = np.array([signal_idx == i for i in gmm.predict(res)])

    # refit the spline on the "signal" peptides
    if refit:
        model = spline(a[signal], formula)

    # calculate new retention times
    o1 = pd.concat((a.reset_index(drop=True),
                    DF({'rt_aligned': np.array(a.rt) - predict(model, rt=a.rt),
                        'status': list(map(lambda x: 'signal' if x else 'noise',
                                           signal))})),
                   axis=1)

    o2 = pd.concat((u.reset_index(drop=True),
                    DF({'rt_aligned': np.array(u.rt) - predict(model, rt=u.rt),
                        'status': 'unlabelled'})),
                   axis=1)

    o = pd.concat((o1, o2))

    if return_model:
        return o, model
    else:
        return o




def denoise_and_align(annotated, unlabelled,
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
        data_frames = workers.starmap(denoise_and_align_run,
                                      iter_groups())

    return pd.concat(data_frames, axis=0)
