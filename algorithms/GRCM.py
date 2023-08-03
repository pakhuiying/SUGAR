import os, glob
import json
from tqdm import tqdm
# import pickle #This library will maintain the format as well
import mutils
import mutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np
from math import ceil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

class GRCM:
    """
    Fell (2022): A Contrast Minimization Approach to Remove Sun Glint in Landsat 8 Imagery
    GRCM: Glint Removal through Contrast Minimisation
    GRCM identifies the entire glint affected area (GAA) by applying an automated analysis of the local reflectance contrast
    Glint mask generated is based on a SWIR band because one of the assumption is that there is no spatial offsets in the channels
    """
    def __init__(self, im_aligned,theta_sol=30):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        """
        self.im_aligned = im_aligned
        self.theta_sol = theta_sol
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]
        self.NIR_band = list(self.wavelength_dict)[-1] #NIR instead of SWIR is used to estimate glint since no SWIR band is available
    
    def get_msk_pgp(self,k_size=3):
        """
        :param thr_pgp (float): threshold THR_PGP
        :param k_size (int): kernel size for MRC
        B7 band (SWIR-2) from OLI's spatial resolution = 30m
        The identification of glint affected pixels and areas is based on the maximum reflectance contrast (MRC),
        defined as a local contrast measure within a 3x3 pixel area
        MRC adopts values greater or equal zero i.e. represents the reflectance contrast between B7 and its darkest neighbouring pixel
        """
        thr_pgp = self.get_thr_pgp(self.theta_sol)
        NIR_im = self.im_aligned[:,:,self.NIR_band]
        MRC = np.zeros(NIR_im.shape)
        nrow, ncol = NIR_im.shape
        k_rad = k_size//2
        for i in range(k_rad,nrow-k_rad):
            for j in range(k_rad,ncol-k_rad):
                dark_pixel = NIR_im[i-k_rad:i+k_rad,j-k_rad:j+k_rad].min()
                MRC[i,j] = NIR_im[i,j] - dark_pixel
        
        MSK_pgp = np.where(MRC>thr_pgp,1,0)
        return MSK_pgp
    
    def get_msk_gap(self, thr_gap=0.2, k_size = 5):
        """
        :param thr_gap (float): threshold THR_GAP to classify as glint affected pixel (GAP)
        :param k_size (int): kernel size for MRC (MxM), where M = 5
        Sun glint affected pixels (GAP) rarely come “alone”, but rather congregate in glint prone areas. 
        This reasoning leads to the following criterion to remove pixels that likely have erroneously been classified as being potentially glinted
        """
        MSK_pgp = self.get_msk_pgp()
        MSK_gap = np.zeros(MSK_pgp.shape)
        nrow, ncol = MSK_pgp.shape
        k_rad = k_size//2
        for i in range(k_rad,nrow-k_rad):
            for j in range(k_rad,ncol-k_rad):
                mean_msk_pgp = MSK_pgp[i-k_rad:i+k_rad,j-k_rad:j+k_rad].mean()
                MSK_gap[i,j] = 1 if mean_msk_pgp > thr_gap else 0

        return MSK_gap
    
    def get_msk_gaa(self, thr_gaa=0.11, k_size=3):
        """
        :param thr_gaa (float): threshold THR_GAA to classify as glint affected area (GAA)
        :param k_size (int): kernel size for MRC (NxN), where N = 3
        A pixel is therefore considered to be located in a GAA if the relative coverage of glint affected pixels within the N x N window centered at position (i, j) exceeds threshold THR
        """
        MSK_gap = self.get_msk_gap()
        MSK_gaa = np.zeros(MSK_gap.shape)
        nrow, ncol = MSK_gap.shape
        k_rad = k_size//2
        for i in range(k_rad,nrow-k_rad):
            for j in range(k_rad,ncol-k_rad):
                mean_msk_gap = MSK_gap[i-k_rad:i+k_rad,j-k_rad:j+k_rad].mean()
                MSK_gaa[i,j] = 1 if mean_msk_gap > thr_gaa else 0

        return MSK_gaa

    def get_thr_pgp(self,theta_sol):
        """
        :param theta_sol (float): solar zenith as deg
        """
        THR_pgp = 0.0005/np.cos(0.95*theta_sol/180*np.pi)
        return THR_pgp