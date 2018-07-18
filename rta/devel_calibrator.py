"""Develop the calibrator."""
%load_ext autoreload
%autoreload 2
%load_ext line_profiler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rta.align.calibrator       import Calibrator, DTcalibrator
from rta.read_in_data           import big_data
from rta.preprocessing          import preprocess
from rta.models.splines.robust  import robust_spline


if __name__ == "__main__":
    folds_no    = 10
    min_runs_no = 5
    annotated_all, unlabelled_all = big_data()
    d = preprocess(annotated_all, min_runs_no,
                   _get_stats = {'retain_all_stats': True})
    c = Calibrator(d, feature='rt', folds_no=folds_no)
    c.fold()
    c.calibrate()
    # less that 1.3 seconds on default params. 
    # c.results[0].plot()
    # parameters = [{"chunks_no": n }for n in range(2,200)]
    # c.calibrate(parameters)
    c.plot()

    m = c.cal_res[0][2]
    m.plot()
    m.cv_stats
    # finish off the collection of stats for purpose of choosing
    # the best models


    dt_cal = Calibrator(d, feature='dt', folds_no=folds_no)
    dt_cal.fold()
    dt_cal.calibrate()
    dt_cal.plot()

    dt_cal.cal_res[10][2].plot()
    # 

    dt_c = DTcalibrator(d, feature='dt', folds_no=folds_no)
    dt_c.fold()
    dt_c.calibrate()
    # less that 1.3 seconds on default params. 
    # c.results[0].plot()
    # parameters = [{"chunks_no": n} for n in range(2,200)]
    # c.calibrate(parameters)
    dt_c.plot()


    m = dt_c.cal_res[0][2]
    m.plot()


c.cal_res

from rta.models.splines.gaussian_mixture import GaussianMixtureSpline
from rta.models.splines.robust import RobustSpline

R1 = c.D[c.D.run == 1].copy()
x = R1.x.values
# x.shape = (x.shape[0], 1)
y = R1.y.values
# y.shape = (y.shape[0], 1)


gms = GaussianMixtureSpline()
gms.fit(x, y, chunks_no = 20)
gms.signal
gms.is_signal(np.array([10, 40]),
              np.array([10, 40]))

x_new = np.array([10, 40, 76, -100, 200, 160])
i = np.searchsorted(gms.x_percentiles, x_new) - 1
indices_out_of_range = np.isin(i, (-1, 19))
signal = np.full(shape = x_new.shape,
                 fill_value = False,
                 dtype = np.bool_)
# for these the anser is fuck off.
signal[~indices_out_of_range]
x_new[~indices_out_of_range]





len(gms.x_percentiles)


gms.x_percentiles
gms = GaussianMixtureSpline()
gms.fit(x, y, chunks_no = 20)
gms.signal

gms.is_signal(np.array([10, 40]),
              np.array([10, 40]))

# Intersection of two normal distributions:
# https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect
m_s, m_n = gms.means[2,]
p_s, p_n = gms.probs[2,]
W, V = gms.variances[2,]
sd_s, sd_n = sqrt(W), sqrt(V)


from math import sqrt, log as l, exp as e

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

class NoRoot(Exception):
    """No roots exception."""
    pass

def roots_of_binomial(A, B, C):
    """Compute the roots of binomial Ax**2+Bx+C.

    Numerically stable.
    W.H. Press, S.A.Teukolsky, W.T. Vetterling, B.P. Flannery,
    Numerical Recipes in C, Second Edition,
    Cambridge University Press,
    Australia 1992, pages 183-184.

    Args:
        A (float): quadratic coefficient.
        B (float): linear coefficient.
        C (float): constant coefficient.

    Returns:
        tuple: Two numbers: the smaller and the bigger root.
    """
    assert A != 0.0, "The degree of the input must equal 2."
    delta = B**2 - 4*A*C
    if delta < 0.0:
        raise NoRoot("The determinant is negative: {}".format(delta))
    elif delta == 0.0:
        return -B/(2.0*A)
    else:
        Q = -(B + sign(B)*sqrt(delta))/2.0
        x1, x2 = Q/A, C/Q 
        return (x1, x2) if x1 <= x2 else (x2, x1)

# roots_of_binomial(1, 0, 1)
# A, B, C = (-1, 0, 1)
# roots_of_binomial(-1, 0, 1)

from math import inf



def signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n):
    """Establish the region where signal dominates noise.

    Assume that both singal and noise follow normal distrubutions,
    weighted by some probabilities, i.e. a gaussian mixture model.
    Find the region where the signal dominates probabilitically over noise.

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
    if m_s == m_n and sd_s == sd_n:
        # these cases require no root finding
        if p_s > p_n:
            # signal dominates over noise
            return (-inf, inf)
        elif p_s < p_n:
            # noise dominates over signal
            return (inf, -inf)
        else:
            raise ValueError("Knife-edge condition: noise and signal parameters are equal. Too dangerous to decide what is noise, what is signal.")
    else:# binomial A x**2 + B x + C
        A = 1.0/sd_n**2 - 1.0/sd_s**2
        if A == 0: # the same as: sd_s == sd_n
            x = (m_s + m_n)/2.0 + sd_s**2 * (l(p_s) - l(p_n))/(m_n-m_s)
            if m_s > m_n:
                return (x, inf)
            elif m_s < m_n:
                return (-inf, x)
        else:# comparing normal log-densities -> find binomial roots
            sum_sds = sd_s + sd_n
            dif_sds = sd_n - sd_s
            B = 2.0*(m_s/sd_s**2 - m_n/sd_n**2)
            C = 2.0*(l(p_s) - l(p_n) + l(sd_n) - l(sd_s)) - B/2.0
            try:
                x = roots_of_binomial(A, B, C)
            except NoRoot:
                if sd_n * p_s > sd_s * p_n:
                    # signal dominates over noise
                    return (-inf, inf)
                elif sd_n * p_s < sd_s * p_n:
                    # noise dominates over signal
                    return (inf, -inf)
                else:
                    raise ValueError("Knife-edge condition: too dangerous to decide what is noise, what is signal.")
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

test_signal_region()


# plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math


def plot_two_components(m_s, m_n, sd_s, sd_n, p_s, p_n,
                        st_cnt = 3,
                        show = True,
                        plt_style = 'dark_background'):
    """Plot the two densities of that form a mixture.

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
    """
    limit = st_cnt*max(sd_s, sd_n)
    x = np.linspace(m_s - limit, m_n + limit, 1000)
    plt.style.use(plt_style)
    plt.plot(x, p_s * norm.pdf(x, m_s, sd_s), c='blue', label='signal')
    plt.plot(x, p_n * norm.pdf(x, m_n, sd_n), c='red', label='noise')
    plt.legend()
    if show:
        plt.show()


plot_two_components(m_s, m_n, sd_s, sd_n, p_s, p_n)



# srs = []
# for i in range(gms.chunks_no):
#     m_s, m_n = gms.means[i,]
#     p_s, p_n = gms.probs[i,]
#     W, V = gms.variances[i,]
#     sd_s, sd_n = sqrt(W), sqrt(V)
#     sr = signal_region(m_s,
#                        m_n,
#                        sd_s,
#                        sd_n,
#                        p_s,
#                        p_n,
#                        simple_calc=True)
#     srs.append(sr)

import numpy as np
from sklearn.mixture import GaussianMixture


class TwoComponentGaussianMixture(GaussianMixture):
    def __init__(self, *args, **kwds):
        super(TwoComponentGaussianMixture,
              self).__init__(n_components=2, *args, **kwds)

    def fit(self, x):
        if len(x.shape) == 1:
            x.shape = x.shape[0], 1
        super(TwoComponentGaussianMixture,
              self).fit(x)

    def _i(self):
        """Get indices of signal and noise.

        Returns:
            tuple: the index of signal and noise.
        """
        # signal = component with smaller variance
        signal_idx, noise_idx = np.argsort(self.covariances_.ravel())
        return [signal_idx, noise_idx]

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

    def signal_region(self):
        """Establish left and right ends of the signal region.

        The signal region is defined as the region where 
        the density of signal times its probability is greater
        than the density of noise times its probability.

        Returns:
            tuple: left and right ends of the signal region. (+∞,-∞) if it is empty.
        """
        
        m_s, m_n = self.means()
        sd_s, sd_n     = self.standard_deviations()
        p_s, p_n = self.probabilities()

        return signal_region(m_s, m_n, sd_s, sd_n, p_s, p_n)

    def plot(self, st_cnt=3):
        """Plot the noise and the signal curves.

        Args:
            st_cnt (float): the number of standard deviations the plot should focus on.
            show (logical): should the plot be immediately shown?
        Returns:
            plt : a plot, if show = False
        """

def fit_2_component_mixture(x, warm_start=False, *args, **kwds):
    gm = TwoComponentGaussianMixture(warm_start=warm_start, *args, **kwds)
    gm.fit(x)
    return gm


gm = fit_2_component_mixture(y)
gm.means()
gm.standard_deviations()
gm.probabilities()
gm.signal_region()


#

rs = RobustSpline()
rs.fit(x,y,chunks_no=20)
rs.x_percentiles

x.shape