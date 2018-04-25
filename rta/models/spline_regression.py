import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix

from rta.models.base_model import Model

class SplineRegression(Model):
    """Virtual class for different spline regressions."""
    def dmatrices(self, formula):
        self.y, self.X = dmatrices(formula, self.data)

    def predict(self, newdata={}, *args, **kwds):
        if isinstance(newdata, dict):
            newdata.update(kwds)
        elif isinstance(newdata, pd.DataFrame):
            newdata.combine_first(pd.DataFrame(kwds))
        spline_filtered_data = dmatrix(self.X.design_info,
                                       data = newdata)
        predictions = np.dot(spline_filtered_data, self.coef)
        return predictions

    def fitted(self):
        spine_base_times_coefs = np.dot(self.X,
                                        self.coef)
        return spine_base_times_coefs

    def __repr__(self):
        return "This is a spline regression."
