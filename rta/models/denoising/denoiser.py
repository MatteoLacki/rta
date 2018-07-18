"""Base Denoiser.

Mateusz Krzysztof Łącki, July 2018.
AG Stefan Tenzer, Universitat Medizin Mainz.
"""

from rta.models.denoising.base_denoiser import BaseDenoiser

class Denoiser(BaseDenoiser):
    def plot(self):
        pass

    def fit(self, x, y, **kwds):
        pass