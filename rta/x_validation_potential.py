def denoise_align_run(run_cnt):
    """Remove noise and align the retention times in a run."""
    a = annotated_slim.loc[annotated_slim.run == run_cnt,]
    u = unlabelled_slim.loc[unlabelled_slim.run == run_cnt,]
	
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
    model = spline(a[signal], formula, cv=)

    a_rt_aligned = np.array(a.rt) - predict(model, rt=a.rt) 
    u_rt_aligned = np.array(u.rt) - predict(model, rt=u.rt) 

    return signal, a_rt_aligned, u_rt_aligned
