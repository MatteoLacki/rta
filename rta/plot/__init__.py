try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def set_plot_options(style="dark_background"):
    """Set plot options.

    Args:
        style (str): The style of the matplotlib visualization.
        Check https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
    """ 
    plt.style.use(style)


if plt:
	plt.style.use('dark_background')
