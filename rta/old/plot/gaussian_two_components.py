"""Plotting functions for 2 component gaussian models."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_two_components(m_s, m_n, sd_s, sd_n, p_s, p_n,
                        st_cnt = 3,
                        show = True,
                        plt_style = 'dark_background'):
    """Plot the two one dimensional densities of that form a mixture.

    Args:
        m_s (float):    the mean of the signal distributions.
        m_n (float):    the mean of the noise distributions.
        sd_s (float):   the standard deviation of the signal distributions.
        sd_n (float):   the standard deviation of the noise distributions.
        p_s (float):    the probability of the signal.
        p_n (float):    the standard deviation of the noise.
        st_cnt (float): the number of standard deviations the plot should focus on.
    Returns:
        plt: some plot object that no-one sees, unless plt.show is executed.
    Examples:
        >>> plot_two_components(0.0, 0.1, 1.0, 0.2, 0.9, 0.1)
        >>> plot_two_components(0.0, 1.0, 1.0, 1.0, 0.3, 0.7)
    """
    limit = st_cnt*max(sd_s, sd_n)
    x = np.linspace(m_s - limit, m_n + limit, 1000)
    plt.style.use(plt_style)
    plt.plot(x, p_s * norm.pdf(x, m_s, sd_s), c='blue', label='signal')
    plt.plot(x, p_n * norm.pdf(x, m_n, sd_n), c='red',  label='noise')
    plt.legend()
    if show:
        plt.show()
