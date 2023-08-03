import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import swirl, resize
from scipy.ndimage import gaussian_filter
from math import ceil
import cv2
class GloriaSimulate:
    """
    simulate different water spectra from different water body types and water types using GLORIA dataset
    Lehmann, Moritz K; Gurlin, Daniela; Pahlevan, Nima; Alikas, Krista; Anstee, Janet M; Balasubramanian, Sundarabalan V; Barbosa, Cláudio C F; Binding, Caren; Bracher, Astrid; Bresciani, Mariano; Burtner, Ashley; Cao, Zhigang; Conroy, Ted; Dekker, Arnold G; Di Vittorio, Courtney; Drayson, Nathan; Errera, Reagan M; Fernandez, Virginia; Ficek, Dariusz; Fichot, Cédric G; Gege, Peter; Giardino, Claudia; Gitelson, Anatoly A; Greb, Steven R; Henderson, Hayden; Higa, Hiroto; Irani Rahaghi, Abolfazl; Jamet, Cédric; Jiang, Dalin; Jordan, Thomas; Kangro, Kersti; Kravitz, Jeremy A; Kristoffersen, Arne S; Kudela, Raphael; Li, Lin; Ligi, Martin; Loisel, Hubert; Lohrenz, Steven; Ma, Ronghua; Maciel, Daniel A; Malthus, Tim J; Matsushita, Bunkei; Matthews, Mark; Minaudo, Camille; Mishra, Deepak R; Mishra, Sachidananda; Moore, Tim; Moses, Wesley J; Nguyen, Hà; Novo, Evlyn M L M; Novoa, Stéfani; Odermatt, Daniel; O'Donnell, David M; Olmanson, Leif G; Ondrusek, Michael; Oppelt, Natascha; Ouillon, Sylvain; Pereira Filho, Waterloo; Plattner, Stefan; Ruiz Verdú, Antonio; Salem, Salem I; Schalles, John F; Simis, Stefan G H; Siswanto, Eko; Smith, Brandon; Somlai-Schweiger, Ian; Soppa, Mariana A; Spyrakos, Evangelos; Tessin, Elinor; van der Woerd, Hendrik J; Vander Woude, Andrea J; Vandermeulen, Ryan A; Vantrepotte, Vincent; Wernand, Marcel Robert; Werther, Mortimer; Young, Kyana; Yue, Linwei (2022)
    : GLORIA - A global dataset of remote sensing reflectance and water quality from inland and coastal waters. PANGAEA, https://doi.org/10.1594/PANGAEA.948492
    :TODO
    """
    def __init__(self,fp_rrs,fp_meta,water_type,sigma=20):
        """
        :param fp_rrs (str): folder path to GLORIA Rrs dataset (multiply by pi to get surface reflectance)
        :param fp_meta (str): folder path to metadata
        :param water_type (list of int): where...
            1: sediment-dominated
            2: chl-dominated
            3: CDOM-dominated
            4: Chl+CDOM-dominated
            5: Moderate turbid coastal (e.g., 0.3<TSS<1.2 & 0.5 <Chl<2.0)
            6: Clear (e.g., TSS<0.3 & 0.1<Chl<0.7)
        """
        self.fp_rrs = fp_rrs
        self.fp_meta = fp_meta
        self.water_type = water_type
        # import wavelengths for each band
        self.wavelength_dict = {5: 444,
                                0: 475,
                                6: 531,
                                1: 560,
                                7: 650,
                                2: 668,
                                8: 705,
                                4: 717,
                                9: 740,
                                3: 842}
        self.n_bands = len(self.wavelength_dict.values())
        self.sigma = sigma

    def import_GLORIA(self):
        """
        import GLORIA datasets as pd dataframe
        """
        df_rrs = pd.read_csv(self.fp_rrs)
        # print(df_rrs.head())
        df_rrs.set_index(df_rrs.columns[:2].to_list(),inplace=True)
        # multiple Rrs by pi to get surface reflectance
        df_rrs[df_rrs.select_dtypes(include=['number']).columns] *= np.pi
        return df_rrs
    
    def import_metadata(self):
        """
        import GLORIA metadata
        """
        df_meta = pd.read_csv(self.fp_meta)
        # filter based on water type
        df_meta = df_meta[df_meta['Water_type'].isin(self.water_type)]
        # drop NA if NAs found in Chl and TSS column
        # df_meta.dropna(subset=['Chla_plus_phaeo', 'TSS'],inplace=True)
        df_meta.set_index(df_meta.columns[0],inplace=True)
        return df_meta
    
    def get_wavelengths(self):
        """
        get the multispectral bands closest to micasense bands
        (GLORIA dataset is recorded at every 1nm interval)
        returns the column index corresponding to the wavelengths in df_rrs
        """
        df_rrs = self.import_GLORIA()
        column_names = list(df_rrs.columns)
        wavelengths = list(self.wavelength_dict.values())
        counter = 0
        column_idx = []
        for i in range(len(column_names)):
            if str(int(wavelengths[counter])) in column_names[i]:
                # print(wavelengths[counter],column_names[i])
                column_idx.append(i)
                counter += 1
            if counter > len(wavelengths) - 1:
                break
        df_rrs = df_rrs.iloc[:,column_idx]
        df_rrs.dropna(inplace=True)
        return df_rrs
    
    def get_filtered_df(self):
        """
        where first 10 columns are Rrs
        """
        df_rrs = self.get_wavelengths()
        df_meta = self.import_metadata()
        # inner join
        df = pd.merge(df_rrs,df_meta,left_index=True,right_index=True)
        # df.dropna(axis='columns',inplace=True)
        # df = df.sort_values(by=['TSS','aCDOM440','Chla_plus_phaeo'])
        # df_list = dict()#{water_type: None for water_type in self.water_type}
        # for water_type in self.water_type:
        #     df_subset = df[df['Water_type']==water_type]
        #     df_list[water_type] = df_subset
        return df#df_list

    def get_rgb_idx(self):
        """
        returns the idx of rgb_bands based on sorted wavelengths
        the idx will be used to identify the corresponding rgb columns from df_rrs
        """
        rgb_bands = [self.wavelength_dict[i] for i in [2,1,0]]
        # print(rgb_bands)
        wavelength_dict = {v:i for i,v in enumerate(self.wavelength_dict.values())}
        # print(wavelength_dict)
        rgb_idx = [wavelength_dict[wavelength] for wavelength in rgb_bands]
        # print(rgb_idx)
        return rgb_idx
    
    def get_image(self,n_rrs=2,nrow=64,ncol=64,scale=5,plot=True,set_seed=False):
        """
        :param n_rrs (int): number of distinct Rrs observation
        :param nrow (int): y dimension of image
        :param ncol (int): x dimension of image
        :param scale (float): scale Rrs by a factor
        :param set_seed (bool): to ensure replicability if needed
        """
        rgb_idx = self.get_rgb_idx()

        df = self.get_filtered_df()
        df_rows = len(df.index)
        print(f'nrows in df: {df_rows}')

        if set_seed is True:
            np.random.seed(1)

        df_idx = np.random.randint(df_rows,size=n_rrs).tolist()
        df_rrs = df.iloc[df_idx,:self.n_bands]
        # convert df to array
        rrs = df_rrs.to_numpy().reshape(1,-1,self.n_bands)
        y,x = rrs.shape[0], rrs.shape[1]
        rep_y = nrow
        rep_x = ceil(ncol/x)

        # note that im is in wavelength order rather than band number order
        im = np.repeat(rrs,rep_x,axis=1)
        im = np.repeat(im,rep_y,axis=0)
        im = im[:nrow,:ncol,:]*scale

        for i in range(im.shape[-1]):
            im[:,:,i] = gaussian_filter(im[:,:,i],sigma=self.sigma)

        if plot is True:
            fig, axes = plt.subplots(1,2)
            rgb_im = np.take(im,rgb_idx,axis=2)
            print(f'Highest Rrs: {rgb_im.max()}')
            rgb_im_norm = cv2.normalize(rgb_im,None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            im_list = [rgb_im,rgb_im_norm]
            title_list = ['RGB','RGB Normalised']
            for img, title, ax in zip(im_list,title_list,axes.flatten()):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            plt.show()

        return im

    def image_distortion(self,im,rotation=0,strength=10,radius=120,plot=True):
        """
        :param rotation (float): Additional rotation applied to the image.
        :param strength (float): The amount of swirling applied.
        :param radius (float): The extent of the swirl in pixels. The effect dies out rapidly beyond radius.
        to simulate non-homogenous background spectra
        https://stackoverflow.com/questions/225548/resources-for-image-distortion-algorithms
        https://scikit-image.org/docs/stable/auto_examples/transform/plot_swirl.html
        """

        rgb_idx = self.get_rgb_idx()
        swirled = swirl(im, rotation=rotation, strength=strength, radius=radius)

        if plot is True:
            fig, axes = plt.subplots(1,2)
            rgb_im = np.take(swirled,rgb_idx,axis=2)
            print(f'Highest Rrs: {rgb_im.max()}')
            rgb_im_norm = cv2.normalize(rgb_im,None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            im_list = [rgb_im,rgb_im_norm]
            title_list = ['RGB','RGB Normalised']
            for img, title, ax in zip(im_list,title_list,axes.flatten()):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            plt.show()

        return swirled

if __name__ == "__main__":
    fp_rrs = os.path.join(os.getcwd(),'GLORIA_2022','GLORIA_Rrs_mean.csv')
    fp_meta = os.path.join(os.getcwd(),'GLORIA_2022','GLORIA_meta_and_lab.csv')
    water_type = [1,5]
    G = GloriaSimulate(fp_rrs,fp_meta,water_type,sigma=10)
    im = G.get_image(n_rrs=4,scale=1,plot=True,set_seed=True)
    im = G.image_distortion(im,rotation=0,strength=10,radius=120,plot=False)
    # im = G.image_distortion(im,rotation=0,strength=0,radius=0,plot=True)
    im = resize(im,(919,1226,10),anti_aliasing=True)
    
    rgb_idx = G.get_rgb_idx()
    # print(rgb_idx)
    plt.figure()
    plt.imshow(np.take(im,rgb_idx,axis=2))
    plt.axis('off')
    plt.show()