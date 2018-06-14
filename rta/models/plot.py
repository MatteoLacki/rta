import numpy as np
import matplotlib.pyplot as plt


def plot_curve(model,
               step = .1,
               out_x_range = False,
               **kwds):
    x_min = min(model.x)
    x_max = max(model.y)
    xs = np.arange(x_min, x_max, step)
    ys = model.predict(xs)
    if 'c' not in kwds:
        kwds['c'] = 'red'
    plt.plot(xs, ys, **kwds)
    if out_x_range:
        return xs


def plot(model,
         step = .1,
         out_x_range = False,
         plt_style = 'dark_background',
         **kwds):
    # TODO: extract the names from design_info.
    plt.style.use(plt_style)
    plt.scatter(model.x,
                model.y,
                s=.4,
                c=model.signal.reshape(-1, 1))
    plot_curve(model, 
               step,
               out_x_range,
               **kwds)
    if out_x_range:
        return x_range
