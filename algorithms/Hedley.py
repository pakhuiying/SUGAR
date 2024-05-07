import os, glob
import json
from tqdm import tqdm
# import pickle #This library will maintain the format as well
import importlib
import radiometric_calib_utils
import mutils
importlib.reload(radiometric_calib_utils)
importlib.reload(mutils)
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

class Hedley:
    def __init__(self, im_aligned,bbox,mode="regression",sigma=2,smoothing=True,glint_mask=True):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): bbox of a glint area e.g. water_glint
        :param mode (str): modes for estimating the slopes for Hedley correction (e.g. regression, least_sq, covariance,pearson)
        :param sigma (float): smoothing sigma for images
        :param smoothing (bool): whether to smooth images or not, due to the spatial offset in glint across diff bands
            The smoothed NIR is used to calculate the regression with other bands, 
            and the smoothed NIR is used to correct other bands
        :param glint_mask (bool): whether to calculate a glint_mask using red to blue ratio
        """
        self.im_aligned = im_aligned
        if mode not in ['regression, least_sq, covariance,pearson']:
            self.mode = 'regression'
        else:
            self.mode = mode
        self.smoothing = smoothing
        self.glint_mask = glint_mask
        self.bbox = mutils.sort_bbox(bbox)
        ((x1,y1),(x2,y2)) = self.bbox
        self.glint_area = self.im_aligned[y1:y2,x1:x2,:]
        if self.smoothing is True:
            self.im_aligned_smoothed = gaussian_filter(self.im_aligned, sigma=sigma)
        self.sigma = sigma
        self.glint_area_smoothed = gaussian_filter(self.glint_area, sigma=self.sigma)
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]
        self.NIR_band = list(self.wavelength_dict)[-1]
    
    # def sort_bbox(self):
    #     ((x1,y1),(x2,y2)) = self.bbox
    #     if x1 > x2:
    #         x1, x2 = x2, x1
    #     if y1 > y2:
    #         y1,y2 = y2, y1

    #     return ((x1,y1),(x2,y2))

    def regression_slope(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope and the intercept, and the model
        """
        lm = LinearRegression().fit(NIR, band)
        b = lm.coef_[0][0]
        intercept = lm.intercept_[0]
        return (b, intercept, lm)
    
    def covariance(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        n = NIR.shape[0]
        pij = np.dot(NIR,band)/n - np.sum(NIR)/n*np.sum(band)/n
        pjj = np.dot(NIR,NIR)/n - (np.sum(NIR)/n)**2
        return pij/pjj
    
    def least_sq(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        A = np.vstack([NIR,np.ones(NIR.shape[0])]).T
        m, _ = np.linalg.lstsq(A,band, rcond=None)[0]
        return m
    
    def pearson(self,NIR,band):
        """
        :param NIR (np.ndarray): flattened np.array for the glint area in NIR band
        :param band (np.ndarray): flattened np.array for the glint area in other band
        returns the slope
        """
        if len(NIR.shape) > 1:
            NIR = NIR.flatten()
            band = band.flatten()
        return pearsonr(NIR,band)[0]
        
    def plot_regression(self):
        """ 
        Construct a linear regression of NIR reflectance versus the reflectance in the ith band using pixels from the deep water subset with glint
        returns a dict of regression slopes in band order i.e. band 0,1,2,3,4,5,6,7,8,9
        """
        #----------------plot rgb before and after correction, and magnified view of bbox----------
        rgb_bands = [2,1,0]
        corrected_bands = self.get_corrected_bands(plot=False)
        rgb_im_corrected = np.stack([corrected_bands[i] for i in rgb_bands],axis=2)
        rgb_im = np.take(self.im_aligned,rgb_bands,axis=2)
        fig, axes = plt.subplots(1,3,figsize=(10,3))
        axes[0].imshow(rgb_im, aspect="auto")
        coord, w, h = mutils.bboxes_to_patches(self.bbox)
        rect = patches.Rectangle(coord, w, h, linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('Original RGB image')
        axes[0].axis('off')

        axes[2].set_title('RGB image after Hedley\'s correction')
        axes[2].imshow(rgb_im_corrected, aspect="auto")
        axes[2].axis('off')

        axes[1].imshow(rgb_im[self.bbox[0][1]:self.bbox[1][1],self.bbox[0][0]:self.bbox[1][0],:], aspect="auto")
        axes[1].set_title('Magnified view of selected glint region')
        axes[1].spines[:].set_linewidth(3)
        axes[1].spines[:].set_color('green')
        axes[1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
        # [i.set_linewidth(3) for i in axes[2].spines[:].itervalues()]
        
        plt.tight_layout()
        plt.show()

        #------------plot regression---------------
        if self.smoothing is True:
            NIR_pixels = self.glint_area_smoothed[:,:,self.NIR_band].flatten().reshape(-1, 1)
        else:
            NIR_pixels = self.glint_area[:,:,self.NIR_band].flatten().reshape(-1, 1)
        self.R_min = np.percentile(NIR_pixels,5,interpolation='nearest')
        nrow = 3
        ncol = 4
        fig, axes = plt.subplots(nrow,ncol,figsize=(10,8))
        for i, ((band_index,wavelength),ax) in enumerate(zip(self.wavelengths,axes.flatten())):
            y = self.glint_area[:,:,band_index].flatten().reshape(-1, 1)
            b_regression, intercept, lm = self.regression_slope(NIR_pixels,y)
            ax.plot(NIR_pixels,y,'.',alpha=0.15)
            ax.set_title(f'{wavelength} nm\n'+r'$\beta =$ {:.3f}, N = {}'.format(b_regression,y.shape[0]))
            ax.set_ylabel(r'$R_T(\lambda)$')
            ax.set_xlabel(r'$R_T(NIR)$')
            # plot regression line
            x_vals = np.linspace(np.min(NIR_pixels),np.max(NIR_pixels),50)
            y_vals = intercept + b_regression * x_vals
            ax.plot(x_vals.reshape(-1,1), y_vals.reshape(-1,1), 'k--',linewidth = 2,label='Regression line')
            # add 1:1 line
            # ax.axline((0, 0), slope=1)
        handles, labels = ax.get_legend_handles_labels()

        del_axes = int(nrow*ncol - len(self.wavelengths))
        for ax in axes.flatten()[-del_axes:]:
            ax.axis('off')
        
        fig.legend(handles=handles,labels=labels,loc='upper center', 
               bbox_to_anchor=(0.55, 0.05),ncol=2,fontsize=12)
        plt.tight_layout()
        plt.show()
        return
    
    def get_glint_mask(self,NIR_threshold=0.8):
        """
        use R/B band ratio to determine glint mask
        NIR_threshold is for red:blue band ratio
        returns a np.ndarray (1 channel)
        """
        r_b_im = self.im_aligned[:,:,2]/self.im_aligned[:,:,0]
        mask = np.where(r_b_im>NIR_threshold,1,0) #where glint pixel is 1, and non-glint pixel is 0
        return mask
    
    def correction_bands(self):
        """ 
        returns a list of slope in band order i.e. 0,1,2,3,4,5,6,7,8,9
        """
        
        if self.smoothing is True:
            NIR_pixels = self.glint_area_smoothed[:,:,self.NIR_band].flatten().reshape(-1, 1)
        else:
            NIR_pixels = self.glint_area[:,:,self.NIR_band].flatten().reshape(-1, 1)
        self.R_min = np.percentile(NIR_pixels,5,interpolation='nearest')

        b_list = []
        for band_number in range(self.n_bands):
            if self.smoothing is True:
                y = self.glint_area_smoothed[:,:,band_number].flatten().reshape(-1, 1)
            else:
                y = self.glint_area[:,:,band_number].flatten().reshape(-1, 1)
            if self.mode == 'regression':
                b,_,_ = self.regression_slope(NIR_pixels,y)
            elif self.mode == 'covariance':
                b = self.covariance(NIR_pixels,y)
            elif self.mode == 'least_sq':
                b = self.least_sq(NIR_pixels,y)
            else:
                b = self.pearson(NIR_pixels,y)
            b_list.append(b)

        return b_list
    
    def get_corrected_bands(self, plot = True):

        ((x1,y1),(x2,y2)) = self.bbox

        b_list = self.correction_bands()

        if self.glint_mask is True:
            gm = self.get_glint_mask()
        hedley_c = lambda x,RT_NIR,b,R_min: x - b*(RT_NIR - R_min)
        # hedley_c = lambda x,RT_NIR,b,R_min: x - b*(RT_NIR - R_min) if (RT_NIR*b > x) else x

        corrected_bands = []
        # avg_reflectance = []
        # avg_reflectance_corrected = []

        fig, axes = plt.subplots(self.n_bands,2,figsize=(10,20))
        for band_number in range(self.n_bands):
            b = b_list[band_number]
            if self.smoothing is True:
                # apply only the blured NIR band only to correct the glint extent for all bands (since here is a spatial discrepancy in glint distribution)
                # but we use the original glint extent for all the bands
                corrected_band = hedley_c(self.im_aligned[:,:,band_number],self.im_aligned_smoothed[:,:,self.NIR_band],b,self.R_min)
                
                # avg_reflectance.append(np.mean(self.glint_area[:,:,band_number]))
                # avg_reflectance_corrected.append(np.mean(corrected_band[y1:y2,x1:x2]))
                axes[band_number,0].imshow(self.im_aligned[:,:,band_number],vmin=0,vmax=1)
            else:
                corrected_band = hedley_c(self.im_aligned[:,:,band_number],self.im_aligned[:,:,self.NIR_band],b,self.R_min)
                
                # avg_reflectance.append(np.mean(self.glint_area[:,:,band_number]))
                # avg_reflectance_corrected.append(np.mean(corrected_band[y1:y2,x1:x2]))
                axes[band_number,0].imshow(self.im_aligned[:,:,band_number],vmin=0,vmax=1)
            
            if self.glint_mask is True:
                cb = self.im_aligned[:,:,band_number].copy()
                cb[gm>0] = corrected_band[gm>0]
            else:
                cb = corrected_band

            corrected_bands.append(cb)

            axes[band_number,1].imshow(cb,vmin=0,vmax=1)
            axes[band_number,0].set_title(f'Band {self.wavelength_dict[band_number]} reflectance')
            axes[band_number,1].set_title(f'Band {self.wavelength_dict[band_number]} reflectance corrected')

        if plot is True:
            for ax in axes.flatten():
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        else:
            plt.close()

        return corrected_bands
    
    def correction_stats(self):
        """
        :param corrected_bands (list of np.ndarrays): images corrected for sunglint
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = self.bbox
        corrected_bands = self.get_corrected_bands(plot=False)
        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        #non-corrected images and reflectance for bbox
        rgb_im = np.take(self.im_aligned,rgb_bands,axis=2)
        avg_reflectance = [np.mean(self.glint_area[:,:,band_number]) for band_number in range(self.n_bands)]
        
        #corrected images and reflectance for bbox
        rgb_im_corrected = np.stack([corrected_bands[i] for i in rgb_bands],axis=2)
        avg_reflectance_corrected = [np.mean(corrected_bands[band_number][y1:y2,x1:x2]) for band_number in range(self.n_bands)]
        
        # plot original rgb
        axes[0,0].imshow(rgb_im)
        axes[0,0].set_title('Original RGB')
        coord, w, h = mutils.bboxes_to_patches(self.bbox)
        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
        axes[0,0].add_patch(rect)
        # plot corrected rgb
        axes[0,1].imshow(rgb_im_corrected)
        axes[0,1].set_title('Corrected RGB')
        rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
        axes[0,1].add_patch(rect)
        # reflectance
        axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)], label='Original')
        axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance_corrected[i] for i in list(self.wavelength_dict)], label='Corrected')
        axes[0,2].set_xlabel('Wavelengths (nm)')
        axes[0,2].set_ylabel('Reflectance')
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title('Corrected and original mean reflectance')

        residual = [og-cor for og,cor in zip(avg_reflectance,avg_reflectance_corrected)]
        axes[0,3].plot(list(self.wavelength_dict.values()),[residual[i] for i in list(self.wavelength_dict)])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Residual in Reflectance')

        for ax in axes[0,:2]:
            ax.axis('off')
        
        h = y2 - y1
        w = x2 - x1

        axes[1,0].imshow(rgb_im[y1:y2,x1:x2,:])
        axes[1,1].imshow(rgb_im_corrected[y1:y2,x1:x2,:])
        axes[1,0].set_title('Original Glint')
        axes[1,1].set_title('Corrected Glint')

        axes[1,0].plot([0,h],[h//2,h//2],color="red",linewidth=3)
        axes[1,1].plot([0,h],[h//2,h//2],color="red",linewidth=3)
        
        # for ax in axes[1,0:2]:
        #     ax.axis('off')

        for i,c in zip(range(3),['r','g','b']):
            axes[1,2].plot(list(range(w)),rgb_im[y1:y2,x1:x2,:][h//2,:,i],c=c)
            # plot for original
        for i,c in zip(range(3),['r','g','b']):
            # plot for corrected reflectance
            axes[1,2].plot(list(range(w)),rgb_im_corrected[y1:y2,x1:x2,:][h//2,:,i],c=c,ls='--')

        axes[1,2].set_xlabel('Width of image')
        axes[1,2].set_ylabel('Reflectance')
        axes[1,2].set_title('Reflectance along red line')

        lines = [Line2D([0], [0], color='black', linewidth=3, linestyle=ls) for ls in ['-','--']]
        labels = ['Original','Corrected']
        axes[1,2].legend(lines,labels,loc='upper right')

        residual = rgb_im[y1:y2,x1:x2,:][h//2,:,:] - rgb_im_corrected[y1:y2,x1:x2,:][h//2,:,:]
        for i,c in zip(range(3),['r','g','b']):
            axes[1,3].plot(list(range(w)),residual[:,i],c=c)

        axes[1,3].set_xlabel('Width of image')
        axes[1,3].set_ylabel('Residual in Reflectance')
        axes[1,3].set_title('Reflectance along red line')

        plt.tight_layout()
        plt.show()

        return
    
    def compare_image(self,save_dir=None, filename = None, plot = True):
        """
        :param save_dir (str): specify directory to store image in. If it is None, no image is saved
        :param filename (str): filename of image u want to save e.g. 'D:\\EPMC_flight\\10thSur24Aug\\F2\\RawImg\\IMG_0192_1.tif'
        returns a figure where left figure is original image, and right figure is corrected image
        """
        corrected_bands = self.get_corrected_bands(plot=False)
        corrected_bands = np.stack(corrected_bands,axis=2)
        corrected_im = np.take(corrected_bands,[2,1,0],axis=2)
        original_im = np.take(self.im_aligned,[2,1,0],axis=2)
        fig, axes = plt.subplots(1,2,figsize=(12,7))
        axes[0].imshow(original_im)
        axes[0].set_title('Original Image'+ r'($\sigma^2_T$' + f': {np.var(self.im_aligned):.4f})')
        # ax.set_title(f'Iter {i} ' + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        axes[1].imshow(corrected_im)
        axes[1].set_title('Corrected Image'+ r'($\sigma^2_T$' + f': {np.var(corrected_bands):.4f})')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()

        if save_dir is not None:
            save_dir = os.path.join(save_dir,"corrected_images")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            filename = mutils.get_all_dir(filename,iter=4)
            filename = os.path.splitext(filename)[0]
            full_fn = os.path.join(save_dir,filename)

            fig.suptitle(filename)
            fig.savefig('{}.png'.format(full_fn))

        if plot is True:
            plt.show()
        else:
            plt.close()
        return