"""Scripts used to save the data, dummy and real."""

import numpy as np
import os


def save_array(filename, x,
               data_path = os.path.join(os.getcwd(), 'rta/data/')):
    """Save an array x.

    Args:
        filename (str):  The name of the file to save 'x' into.
        x (np.array):    Values to save.
        data_path (str): Folder where the file should be saved.
    """ 
    np.save(file=os.path.join(data_path, filename), arr=x)


def save_spectrum(mz, intensity,
                  mz_filename        = 'mz.npy', 
                  intensity_filename = 'intensity.npy',
                  data_path          = os.path.join(os.getcwd(), 'rta/data/')):
    """Save a spectrum consisting of m/z and intensities arrays.

    Args:
        mz (np.array):            Mass to charge values to save.
        intensity (np.array):     Intensities corrresponding to mass/charge values.
        mz_filename (str):        The name of the file to save m/z into.
        intensity_filename (str): The name of the file to save m/z into.
        data_path (str):          Folder where the files should be saved.
    """ 
    save_array(mz_filename,         mz,        data_path)
    save_array(intensity_filename,  intensity, data_path)