%load_ext autoreload
%autoreload 2
%load_ext line_profiler


import numpy as np
from rta.models.robust_spline import RobustSpline

import pandas as pd
import matplotlib.pyplot as plt

model = RobustSpline()
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
chunks_no = 3
model.adjust(x, y)
model.fit(x, y, chunks_no=3)



from rta.array_operations.misc import percentiles
x_inner_percentiles = percentiles(x, 8, inner=True)




from rta.models.GMLSQSpline import fit_spline

splajn = fit_spline(x, y, 4)

x_plot = np.linspace(min(x), max(x), 100)
y_plot = splajn(x_plot)


plt.plot(x_plot, y_plot, c='orange')
plt.scatter(x, y)
plt.show()





def test_RobustSpline():


