import matplotlib.pyplot as plt



def plot_runs(data,
              runs_to_avoid = [],
              run_2_name = {},
              run_name  = 'run',
              x         = 'runs_stat_0',
              y         = 'rt_0',
              plt_style = 'dark_background',
              title     = '',
              show      = True):
    """Plot distnces to medians for annotated peptides."""
    plt.style.use(plt_style)
    for r, d in data.groupby('run'):
        if r not in runs_to_avoid:
            x = d.runs_stat_0
            y = d.rt_0 - x
            label = run_2_name[r] if run_2_name else str(r)
            plt.scatter(x, y, label=label, s=.4)
            if title:
                plt.title(title)
    plt.legend(markerscale = 4)
    if show:
        plt.show()


def plot_experiment_comparison(datasets,
                               titles    = None,
                               show      = True, 
                               plt_style = 'dark_background',
                             **all_plots_settings):
    K = len(datasets)
    datasets = datasets.__iter__()
    i = 1
    d = next(datasets)
    first_plot = plt.subplot(K,1,i)
    plot_runs(d,
              show      = False,
              plt_style = plt_style,
              title     = titles[i-1] if titles is not None else '',
              **all_plots_settings)
    for e in datasets:
        i += 1
        plt.subplot(K,1,i, 
                    sharex = first_plot,
                    sharey = first_plot)
        plot_runs(e,
                  show      = False,
                  plt_style = plt_style,
                  title     = titles[i-1] if titles is not None else '',
                 **all_plots_settings)
    if show:
        plt.show()

