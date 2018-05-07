from rta.models.least_squares_splines import least_squares_spline
from rta.models.huber import huber_spline
from rta.models.RANSAC import RANSAC_spline
from rta.models.theil_sen import theil_sen_spline
from rta.models.quantile import quantile_spline
from rta.models.base_model import predict, fitted, coef, residuals

def spline(data, formula, model='Huber', **kwds):
    """Make spline regression.

    Parameters
    ==========
    data : pandas
        The data to be analyzed. It will not get copied.
    formula :

    """
    splines = {'Theil-Sen': theil_sen_spline,
               'Huber': huber_spline,
               'RANSAC': RANSAC_spline,
               'quantile': quantile_spline,
               'least_squares': least_squares_spline}
    assert model in splines, "God will cut you down. Repent, repent, repent."
    return splines[model](data, formula, **kwds)
