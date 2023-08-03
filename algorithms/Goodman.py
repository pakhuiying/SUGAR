import os, glob
import json
from tqdm import tqdm
# import pickle #This library will maintain the format as well
import importlib
import radiometric_calib_utils
import mutils
importlib.reload(radiometric_calib_utils)
importlib.reload(mutils)
import radiometric_calib_utils as rcu
import mutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np
from math import ceil

class Goodman:
    def __init__(self, im_aligned, NIR_lower = 7, NIR_upper = 9, A = 0.000019, B = 0.1):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param NIR_lower (int): band index which corresponds to 650nm, closest band to 640nm
        :param NIR_upper (int): band index which corresponds to 740nm, closest band to 750nm
        :param A (float): the values in Goodman et al's paper, using AVIRIS reflectance (rather than radiance) data
        :param B (float): the values in Goodman et al's paper, using AVIRIS reflectance (rather than radiance) data
            see Goodman et al, which corrects each pixel independently. The NIR radiance is subtracted from the radiance at each wavelength,
            but a wavelength-independent offset is also added. 
            it is not clear how A and B were chosen, but an optimization for a case where in situ data is
            available would enable values to be found
        """
        self.im_aligned = im_aligned
        self.NIR_lower = NIR_lower
        self.NIR_upper = NIR_upper
        self.A = A
        self.B = B
        self.n_bands = im_aligned.shape[-1]
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}

    def get_corrected_bands(self):
        goodman = lambda r,r_640,r_750, A, B: r - r_750 + (A+B*(r_640-r_750))
        corrected_bands = []
        for i in range(self.n_bands):
            R = self.im_aligned[:,:,i]
            R_640 = self.im_aligned[:,:,self.NIR_lower]
            R_750 = self.im_aligned[:,:,self.NIR_upper]
            corrected_band = goodman(R,R_640,R_750,self.A,self.B)
            corrected_bands.append(corrected_band)
        return corrected_bands
    