"""The Robust Spline class 2: the final countdown.

Here I trykb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rta.array_operations.misc import overlapped_percentile_pairs
from rta.models.denoising.window_based import sort_by_x
from rta.models.splines.spline import Spline
from rta.models.splines.beta_splines import beta_spline