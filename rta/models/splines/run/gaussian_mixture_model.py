"""How to run gaussian mixture.

Run the script from the bottom level with the Makefile.
"""

import numpy as np
import os

from rta.models.splines.gaussian_mixture import GaussianMixtureSpline

def load_arr(name):
    data_path = os.path.join(os.getcwd(), 'rta/data/')
    return np.load(file=os.path.join(data_path, name))

if __name__ == "__main__":
    x = load_arr('x.npy')
    y = load_arr('y.npy')
    gms = GaussianMixtureSpline(chunks_no = 20, warm_start = False)
    gms.fit(x, y)
    gms.plot()
