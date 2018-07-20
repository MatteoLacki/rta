"""Ordered code for the two component guassian mixture model."""
from math import inf, log as l, exp as e, pi
import numpy as np
from sklearn.mixture import GaussianMixture

from rta.math.binomial import binomial_roots
from rta.plotters.gaussian_two_components import plot_two_components


class TwoComponentGaussianMixture(GaussianMixture):
    def __init__(self, *args, **kwds):
        super().__init__(n_components=2, *args, **kwds)

    def fit(self, x):
        """Fit the two component gaussian mixture model.

        Args:
            x (np.array of floats): the data to be grouped.
        """
        if len(x.shape) == 1:
            x.shape = x.shape[0], 1
        super().fit(x)

    def _i(self):
        """Get indices of signal and noise.

        Returns:
            tuple: the index of signal and noise.
        """
        # signal = component with smaller variance
        signal_idx, noise_idx = np.argsort(self.covariances_.ravel())
        return [signal_idx, noise_idx]

    def is_signal_gmm(self, new_x):
        """Is the new point noise or signal.

        Procedure uses the evalutation of the normal densities.
        This can be done once much faster by estimating (once)
        the region where signal's generalized denstity is lower
        than that of noise. 
        Generalized density = density x mixture probability.
        So, better use the 'is_signal' method.

        new_x (np.array of floats): points to classify as signal (True) or noise (False).
        """
        signal_idx,_ = self._i()
        if len(new_x.shape) == 1:
            new_x.shape = new_x.shape[0], 1
        return self.predict(new_x) == signal_idx

    def is_signal(self, new_x):
        """Is the new point noise or signal.

        new_x (np.array of floats): points to classify as signal (True) or noise (False).
        """
        signal_idx,_ = self._i()
        if len(new_x.shape) == 1:
            new_x.shape = new_x.shape[0], 1
        return self.predict(new_x) == signal_idx

    def means(self):
        """Get the estimated means of the signal and noise.

        Returns:
            tuple: the mean of signal and noise distributions.
        """
        i = self._i()
        return self.means_.ravel()[i]

    def probabilities(self):
        """Get the estimated probabilities of the signal and noise.

        Returns:
            tuple: the probability of signal and noise.
        """
        i = self._i()
        return self.weights_[i]

    def standard_deviations(self):
        """Get the estimated standard deviations of the signal and noise distributions.

        Returns:
            tuple: the standard deviation of signal and noise distributions.
        """
        i = self._i()
        return np.sqrt(self.covariances_.ravel()[i])

    def signal_region(self, approximate=False):
        """Establish left and right ends of the signal region.

        The signal region is defined as the region where 
        the density of signal times its probability is greater
        than the density of noise times its probability.

        Args:
            approximate (logical): Should we return the approximate region, which can be only smaller.

        Returns:
            tuple: left and right ends of the signal region. (+∞,-∞) if it is empty.
        """
        m_s, m_n   = self.means()
        sd_s, sd_n = self.standard_deviations()
        p_s, p_n   = self.probabilities()
        if approximate:
            return approximate_signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n)
        else:
            return signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n)

    def plot(self, st_cnt=3, show=True, plt_style='dark_background'):
        """Plot the noise and the signal curves.

        Args:
            st_cnt (float):  the number of standard deviations the plot should focus on.
            show (logical):  should the plot be immediately shown?
            plt_style (str): the style of the pyplot.
        Returns:
            plt : a plot, if show = False, otherwise nothing.
        """
        m_s, m_n   = self.means()
        sd_s, sd_n = self.standard_deviations()
        p_s, p_n   = self.probabilities()
        plot_two_components(m_s, m_n, sd_s, sd_n, p_s, p_n,
                            st_cnt = show,
                            show = show,
                            plt_style = plt_style)



def fit_2_component_mixture(x, warm_start=False, *args, **kwds):
    """Fit the two component gaussian mixture model.

    Args:
        x (np.array of floats): the data to be grouped.
        warm start (logical): if fitting multiple times, should fitting start from previously established solutions?
        *args: parameters for sklearn's gaussian mixture model.
        **kwds: named parameters for sklearn's gaussian mixture model.
    
    .. Link to Sklearn Gaussian Mixtures model:
    http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
    """
    gm = TwoComponentGaussianMixture(warm_start=warm_start, *args, **kwds)
    gm.fit(x)
    return gm


class KnifeEdgeError(Exception):
    """A knife-edge exception."""
    pass


def signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n):
    """Establish the region where signal dominates noise.

    Assume that both singal and noise follow normal distrubutions,
    weighted by some probabilities, i.e. a gaussian mixture model.
    Find the region where the signal dominates probabilitically over noise.
    Calculations here are similar to those in:
    # https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect
    However, our calculations include the probabilities of the two components.

    Args:
        m_s (float): the mean of the signal distributions.
        m_n (float): the mean of the noise distributions.
        sd_s (float):   the standard deviation of the signal distributions.
        sd_n (float):   the standard deviation of the noise distributions.
        p_s (float):    the probability of the signal.
        p_n (float):    the standard deviation of the noise.

    Returns:
        tuple: left and right end of the region where signal dominates probabilistically.

    Raises:
        KnifeEdgeError: if there is no way to attribute a region to noise or signal.
        ValueError: if something stupid happened. Not sure what. It didn't occur yet.
    """
    if m_s == m_n and sd_s == sd_n:
        # these cases require no root finding
        if p_s > p_n:
            # signal dominates over noise
            return (-inf, inf)
        elif p_s < p_n:
            # noise dominates over signal
            return (inf, -inf)
        else:
            raise KnifeEdgeError("Noise and signal parameters are equal: no way to tell apart noise from signal.")
    else:# binomial A x**2 + B x + C
        A = 1.0/sd_n**2 - 1.0/sd_s**2
        if A == 0: # the same as: sd_s == sd_n
            x = (m_s + m_n)/2.0 + sd_s**2 * (l(p_s) - l(p_n))/(m_n-m_s)
            if m_s > m_n:
                return (x, inf)
            elif m_s < m_n:
                return (-inf, x)
            else:
                raise ValueError("Impossible to be here. Hahaha.")
        else:# comparing normal log-densities -> find binomial roots
            sum_sds = sd_s + sd_n
            dif_sds = sd_n - sd_s
            B = 2.0*(m_s/sd_s**2 - m_n/sd_n**2)
            C = 2.0*(l(p_s) - l(p_n) + l(sd_n) - l(sd_s)) - ((m_s/sd_s)**2 - (m_n/sd_n)**2)/2.0
            try:
                x = binomial_roots(A, B, C)
            except NoRootError:
                if sd_n * p_s > sd_s * p_n:
                    # signal dominates over noise
                    return (-inf, inf)
                elif sd_n * p_s < sd_s * p_n:
                    # noise dominates over signal
                    return (inf, -inf)
                else:
                    raise KnifeEdgeError("Noise and signal parameters are equal: no way to tell apart noise from signal.")
            if len(x) == 2:
                return x
            elif len(x) == 1:
                x = x[0]
                if m_s > m_n:
                    return (x, inf)
                elif m_s < m_n:
                    return (-inf, x)
                else:
                    raise ValueError("Impossible that under equal means there is only one point of equal denisty.")
            else:
                raise ValueError("It's impossible to get here.")


def test_signal_region():
    # equally probable densities with the same standard deviations
    # should have equal densities only precisely between the two modes.
    m_s, sd_s, p_s = 0.0, 1.0, 0.5
    m_n, sd_n, p_n = 1.0, 1.0, 0.5
    L, R = signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n)
    assert L == -inf and R == .5

    # signal density entirely dominating the noise density:
    m_s, sd_s, p_s = 0.0, 1.0, 0.9
    m_n, sd_n, p_n = 0.1, 0.2, 0.1
    L, R = signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n)
    assert L == -inf and R == inf


def approximate_signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n):
    """Establish the region where signal dominates noise.

    Assume that both singal and noise follow normal distrubutions,
    weighted by some probabilities, i.e. a gaussian mixture model.
    Find the approximate region where the signal dominates probabilitically over noise.
    The approximation: the region should correspond to finding the two points,
    at which the signal intensity equals the height of the noise peak
    evaluated at the mode of the signal distribution.

    Args:
        m_s (float): the mean of the signal distributions.
        m_n (float): the mean of the noise distributions.
        sd_s (float):   the standard deviation of the signal distributions.
        sd_n (float):   the standard deviation of the noise distributions.
        p_s (float):    the probability of the signal.
        p_n (float):    the standard deviation of the noise.

    Returns:
        tuple: left and right end of the region where signal dominates probabilistically.
    """
    # evaluate the noise log-distribution at signal's mode: neglect 2pi factor.
    noise_level = l(p_n) - 0.5*l(sd_n) - 0.5 * (m_s - m_n)**2 / sd_n**2
    A =  1.0
    B = -2.0 * m_s
    C =  m_s **2 + 2*sd_s**2 * (l(sd_s) + noise_level - l(p_s))
    try:
        x = binomial_roots(A, B, C)
    except NoRootError:
        # All is noise.
        return (inf, -inf)
    if len(x) == 1:
        # All is noise.
        return (inf, -inf)
    else:
        return x