%load_ext autoreload
%autoreload 2
%load_ext line_profiler


import numpy as np
from rta.models.robust_spline import RobustSpline, mad_window_filter

import pandas as pd
import matplotlib.pyplot as plt

model = RobustSpline()
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y = np.array([1.1, 1.2, 1.15, 1.54, 1.7, 1.543, 1.67, 1.64, 1.725,  1.6454,  2.11,  1.98,  2.12])
chunks_no = 3


z = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

model.fit(x, z, chunks_no=3)
mad_window_filter(x, y, chunks_no, 100)


from rta.models.plot import plot

plot(model)
plt.show()

from rta.stats.stats import mad

mad(y, return_median=True)
# write a test for the cv



# interesting observation: 
#   this behaviour is perfeclty normal
#   as the distribution is almost entirely based on two values.






from rta.models.splines.beta_splines import beta_spline

splajn = beta_spline(x, y, 4)

x_plot = np.linspace(min(x), max(x), 100)
y_plot = splajn(x_plot)

plt.plot(x_plot, y_plot, c='orange')
plt.scatter(x, y)
plt.show()





def test_RobustSpline():
    """Check if the algorithm replicates the old results."""

