"""Base Denoiser.

Mateusz Krzysztof Łącki, July 2018.
AG Stefan Tenzer, Universitat Medizin Mainz.
"""

import pandas as pd

def sort_by_x(x, y):
    """Sort x and y by increasing values of x.

    Args:
        x (np.array or list): comparable values.
        x (np.array or list): some values.

    Returns:
        tuple of np.arrays: x and y sorted w.r.t. x.
    """
    d = pd.DataFrame({'x':x, 'y':y})
    d = d.drop_duplicates(subset='x', keep=False)
    d = d.sort_values(['x'])
    return d.x.values, d.y.values


class WindowDenoiser(object):
    def fit(self, x, y,
            chunks_no = 20,
            sort      = False,
            **window_denoising_args):
        """Windowing based filtering.

        Divides m/z values and intensities into chunks.
        For each chunk, estimate the boundary between noise and signal.
        The estimation applies a sliding window approach based on 3 chunks.
        For example, if chunks_no = 5 and E stands for set that takes part in estimation, F is the 
        set on which we fit, and N is a set not taken into consideration,
        then subsequent fittings for 5-chunk division look like this:
        FENNN, EFENN, NEFEN, NNEFE, NNNEF.

        Args:
            x (np.array) 1D control
            y (np.array) 1D response
            chunks_no (int) The number of quantile bins.
            sort (logical) Sort 'x' and 'y' with respect to 'x'?
            window_denoising_args (key words): arguments for the window denoising method.

        Returns:
            signal (np.array of logical values) Is the point considered to be a signal?
            medians (np.array) Estimates of medians in consecutive bins.
            stds (np.array) Estimates of standard deviations in consecutive bins.
            x_percentiles (np.array) Knots of the spline fitting, needed to filter out noise is 'is_signal'.        
        """
        self.x, self.y = sort_by_x(x, y) if sort else (x, y)
        self.signal = np.empty(len(x), dtype=np.bool_)

    def window_denoising(self, **kwds):
        """A function applied for denoising on a particular window."""
        raise NotImplementedError

    def predict(self, new_x, new_y):
        """Predict if a point is noise or not."""
        raise NotImplementedError

    def plot(self):
        """Plot the model."""
        raise NotImplementedError
