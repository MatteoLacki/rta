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
            # x = d.runs_stat_0
            x = d.rt_0
            y = d.rt_0 - d.runs_stat_0
            # y = d.rt_0 - x
            label = run_2_name[r] if run_2_name else str(r)
            plt.scatter(x, y, label=label, s=.4)
            if title:
                plt.title(title)
    plt.legend(markerscale = 4)
    if show:
        plt.show()


def plot_experiment_comparison(results,
                               projects,
                               show      = True, 
                               plt_style = 'dark_background'):
    K = len(projects)
    i = 0
    first_plot = plt.subplot(K,1,i+1)
    c, run_2_name, project, title = results[projects[i]]
    plot_runs(c.D, 
              title         = title,
              run_2_name    = run_2_name,
              show          = False,
              plt_style     = plt_style)
    for e in range(1, K):
        i += 1
        plt.subplot(K,1,i+1, 
                    sharex = first_plot,
                    sharey = first_plot)
        c, run_2_name, project, title = results[projects[i]]
        plot_runs(c.D, 
                  title         = title,
                  run_2_name    = run_2_name,
                  show          = False,
                  plt_style     = plt_style)
    if show:
        plt.show()


def plot_runs_individually(D, runs, i_max, j_max,
                           run_2_name = {},
                           plt_style  = 'dark_background',
                           show       = True):
    DG = D.groupby('run')
    plots_no = len(runs)
    data_iter = enumerate(DG.__iter__())
    def single_plot(I, d, r,
                    plot_axes  = None,
                    size       = .4):
        if I == 0:
            plot_axes = plt.subplot(i_max, j_max, I + 1)
        else:
            plt.subplot(i_max, j_max, I + 1,
                        sharex = plot_axes,
                        sharey = plot_axes)
        x = d.runs_stat_0
        y = d.rt_0 - x
        title = run_2_name[r] if run_2_name else str(r)
        plt.style.use(plt_style)
        plt.scatter(x, y, s=size)
        plt.title(title)
        if I == 0:
            return plot_axes

    I, (r, d) = next(data_iter)
    plot_axes = single_plot(I, d, r)
    for I, (run, d) in data_iter:
        single_plot(I, d, r, plot_axes)
    if show:
        plt.show()