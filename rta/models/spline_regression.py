import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix

from rta.models.base_model import Model

class SplineRegression(Model):
    """Virtual class for different spline regressions."""
    def predict(self, newdata={}, *args, **kwds):
        if isinstance(newdata, dict):
            newdata.update(kwds)
        elif isinstance(newdata, pd.DataFrame):
            newdata.combine_first(pd.DataFrame(kwds))
        # WARNING
        # Spline basis is set to interpolate on a given interval [s, e].
        # As such, it is not obvious what to do outside that interval,
        # so that the the extrapolation is not supported out-of-the-box,
        #
        # For more elaborate methods, such as Gaussian Processes, this is not
        # the case.
        #
        # TODO: investigate the natural-splines and the smoothing-splines.
        spline_filtered_data = dmatrix(self.X.design_info,
                                       data = newdata)
        spline_filtered_data = np.asarray(spline_filtered_data, dtype=np.float64)

        predictions = np.dot(spline_filtered_data, self.coef)
        return predictions

    def fitted(self):
        spine_base_times_coefs = np.dot(self.X, self.coef)
        return spine_base_times_coefs

    @property
    def res(self):
        """Get residuals."""
        return self.y.ravel() - self.fitted()

    def __repr__(self):
        return "This is spline regression."
