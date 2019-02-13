from rta.models.model import Model


class Spline(Model):
    def __call__(self, x):
        """Predict the values at the new data points.

        Args:
            x (np.array): The control variable.
        Returns:
            np.array : Predicted values corresponding to values of 'x'.
        """
        return self.spline(x)