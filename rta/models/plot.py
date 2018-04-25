import numpy as np
import matplotlib.pyplot as plt

def plot(model,
         x_name='rt',
         y_name='rt_median_distance',
         step = .1,
         out_x_range = False,
         **kwds):
    # TODO: extract the names from design_info.
    x_min = min(model.data[x_name])
    x_max = max(model.data[x_name])
    x_range = np.arange(x_min, x_max, step)
    prediction = model.predict(newdata = {x_name: x_range})
    plt.style.use('dark_background')
    plt.scatter(model.data[x_name],
                model.data[y_name],
                s=.4)
    if 'c' not in kwds:
        kwds['c'] = 'red'
    plt.plot(x_range,
             prediction,
             **kwds)
    if out_x_range:
        return x_range
