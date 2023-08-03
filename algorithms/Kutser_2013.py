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
from scipy.optimize import curve_fit

class Kutser:
    def __init__(self, im_aligned, r_lower = [0], r_upper = [8,9]):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param r_lower (list of int): band indices which corresponds to the smallest wavelengths (444nm)
        :param r_upper (list of int): band indices which corresponds to the largest wavelengths (740nm and 842nm)
            see Kutser et al (2013), which corrects each pixel independently. The main advantage of the method is its simplicity. 
            The reflected component can be estimated from each single reflectance spectrum
            itself and no auxiliary data is needed. 
        Note: Kutser is more suitable for correcting signals instead of images pixel-by-pixel as it is very inefficient
        Sometimes best fit parameters cannot be found via optimisation. 
        Each pixel needs to go through the curve fitting optimization process which is very slow
        """
        self.im_aligned = im_aligned
        self.r_lower = r_lower
        self.r_upper = r_upper
        self.n_bands = im_aligned.shape[-1]
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}

    def get_reflectance_by_wavelength(self):
        R = np.take(self.im_aligned,list(self.wavelength_dict),axis=2)
        return R

    def extract_extreme_wavelengths(self,R):
        """ 
        :param R (np.ndarray): ordered reflectance by wavelength
        returns a tuple of (cropped_wavelengths (list of float), cropped_R (np.ndarray))
        """
        idx = self.r_lower + self.r_upper
        cropped_R = np.take(R,idx,axis=2)
        cropped_R = cropped_R.reshape(-1,len(idx))
        sorted_wavelengths = list(self.wavelength_dict.values())
        cropped_wavelengths = [sorted_wavelengths[i] for i in idx]
        return cropped_wavelengths,cropped_R
    
    def get_corrected_bands(self):
        nrow, ncol, n_bands = self.im_aligned.shape
        power_fun = lambda x,a,b: a*x**-b

        R = self.get_reflectance_by_wavelength()
        cropped_wavelengths,cropped_R = self.extract_extreme_wavelengths(R)

        coeff_list = []
        for i in range(cropped_R.shape[0]):
            try:
                popt, _ = curve_fit(power_fun, cropped_wavelengths, cropped_R[i,:],p0=[50,1],maxfev=50)
                coeff_list.append(popt)
            except:
                coeff_list.append(np.array([0,0]))

        A = np.array([i[0] for i in coeff_list])#.reshape(nrow,ncol)
        B = np.array([i[1] for i in coeff_list])#.reshape(nrow,ncol)

        y_fit = np.zeros((nrow*ncol,n_bands))
        sorted_wavelengths = list(self.wavelength_dict.values())
        for i in range(A.shape[0]):
            y_fit[i,:] = power_fun(sorted_wavelengths, A[i],B[i])
        
        y_fit.reshape(self.im_aligned.shape)
        corrected_bands = self.im_aligned - y_fit
        return corrected_bands