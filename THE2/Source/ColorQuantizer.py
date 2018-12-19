#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from scipy.ndimage import imread

from scipy import misc
from PIL import Image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import numpy as np

class ColorQuantizer:
    """Quantizer for color reduction in images. Use MyKMeans class that you implemented.
    
    Parameters
    ----------
    n_colors : int, optional, default: 64
        The number of colors that wanted to exist at the end.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Read more from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    """
    
    def __init__(self, n_colors=64, random_state=None):

        path
    
    def read_image(self, path):
        """Reads jpeg image from given path as numpy array. Stores it inside the
        class in the image variable.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        """
        #self.image = load_sample_image(path)

        self.image = misc.imread(path)

    
    def recreate_image(self, path_to_save):
        """Reacreates image from the trained MyKMeans model and saves it to the
        given path.
        
        Parameters
        ----------
        path_to_save : string, path of the png image to save
        """
        pass

    def export_cluster_centers(self, path):
        """Exports cluster centers of the MyKMeans to given path.

        Parameters
        ----------
        path : string, path of the txt file
        """
        pass
        
    def quantize_image(self, path, weigths_path, path_to_save):
        """Quantizes the given image to the number of colors given in the constructor.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        weigths_path : string, path of txt file to export weights
        path_to_save : string, path of the output image file
        """
        
        """ def quantize(raster, n_colors):
            width, height, depth = raster.shape
            reshaped_raster = np.reshape(raster, (width * height, depth))

            model = cluster.KMeans(n_clusters=n_colors)
            labels = model.fit_predict(reshaped_raster)
            palette = model.cluster_centers_

            quantized_raster = np.reshape(
                palette[labels], (width, height, palette.shape[1]))

            return quantized_raster """

        pass
        

if __name__=='__main__':
    path ="/mnt/d/DERS/0000GITHUB/Machine-Learning/THE2/Docs/ankara.jpg"
    image = ColorQuantizer()
    image.read_image(path)