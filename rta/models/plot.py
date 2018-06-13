import numpy as np
import matplotlib.pyplot as plt


def plot_curve(model,
               x_name='rt',
               y_name='rt_median_distance',
               step = .1,
               out_x_range = False,
               **kwds):
    x_min = min(model.data[x_name])
    x_max = max(model.data[x_name])
    x_range = np.arange(x_min, x_max, step)
    prediction = model.predict(newdata = {x_name: x_range})
    if 'c' not in kwds:
        kwds['c'] = 'red'
    plt.plot(x_range,
             prediction,
             **kwds)
    if out_x_range:
        return x_range


def plot(model,
         step = .1,
         out_x_range = False,
         plt_style = 'dark_background',
         **kwds):
    # TODO: extract the names from design_info.
    plt.style.use(plt_style)
    plt.scatter(model.control,
                model.response,
                s=.4,
                c=model.signal)
    plot_curve(model, 
               model.control_name,
               model.response_name,
               step,
               out_x_range,
               **kwds)
    if out_x_range:
        return x_range
