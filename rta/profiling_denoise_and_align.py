%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import numpy as np
import pandas as pd
from collections import Counter

from rta.read_in_data import big_data
from rta.preprocessing import preprocess
from rta.splines.denoising2 import denoise, denoise_and_align_run

annotated, unlabelled = big_data(path = "~/Projects/retentiontimealignment/Data/")
annotated, annotated_stats = preprocess(annotated, min_runs_no = 2)
annotated_slim = annotated[['run', 'rt', 'rt_median_distance']]
unlabelled_slim = unlabelled[['run', 'rt']]


formula = "rt_median_distance ~ bs(rt, df=40, degree=2, lower_bound=0, upper_bound=200, include_intercept=True) - 1"
model_name = 'Huber'
refit = True
workers_cnt = 4


# def iter_groups():
#         for run_no, a in annotated.groupby('run'):
#             u = unlabelled[unlabelled.run == run_no]
#             yield a, u, formula, model_name, refit

# a, u, formula, model_name, refit = next(iter_groups())
# signal, a_rt_aligned, u_rt_aligned = denoise_and_align_run(a, u, formula, model_name, True)
# type(signal)
# type(a_rt_aligned)
# type(np.asarray(a_rt_aligned))

# %lprun -f denoise_and_align_run denoise_and_align_run(a, u, formula, model_name, refit)

res = denoise(annotated_slim,
			  unlabelled_slim, 
			  formula, 
			  model_name, 
			  refit, 
			  workers_cnt)

res[0][0].shape
res[0][0][1:200]

%lprun -f denoise denoise(annotated, unlabelled, formula, model_name, refit, workers_cnt)
%lprun -f denoise denoise(annotated_slim, unlabelled_slim, formula, model_name, refit, workers_cnt)
%lprun -f denoise denoise(annotated_slim, unlabelled_slim, formula, model_name, refit, 2)



from patsy import dmatrices, dmatrix
from patsy import bs, cr

%%timeit
y, X = dmatrices(formula, annotated_slim)

min_rt = min(min(annotated_slim.rt), min(unlabelled_slim.rt))
max_rt = max(max(annotated_slim.rt), max(unlabelled_slim.rt))
deciles = np.percentile(a.rt, q = range(0,100,10))

bsrt = bs(a.rt, 
		   df=40, 
		   degree=2, 
		   lower_bound=min_rt, 
		   upper_bound=max_rt, 
		   include_intercept=True)


crrt = cr(a.rt, 
		   knots = deciles,
		   lower_bound=min_rt, 
		   upper_bound=max_rt)










from rta.models.huber import huber_spline

HuberModel = huber_spline(annotated_slim, formula)


%%timeit
HuberModel.predict(rt = annotated_slim.rt)

import inspect
print(inspect.getsource(dmatrices))



%lprun -f HuberModel.predict HuberModel.predict(rt = annotated_slim.rt)
%lprun -f HuberModel.predict HuberModel.predict(rt = unlabelled_slim.rt)

def predict(coef, newdata = {}, **kwds):
	if isinstance(newdata, dict):
            newdata.update(kwds)
    elif isinstance(newdata, pd.DataFrame):
        newdata.combine_first(pd.DataFrame(kwds))
	spline_filtered_data = dmatrix(X.design_info,								   
								   data = newdata)
    spline_filtered_data = np.asarray(spline_filtered_data, 
                                      dtype=np.float64)
    predictions = np.dot(spline_filtered_data, coef)
    return np.asarray(predictions)

predict(rt = annotated_slim.rt)
