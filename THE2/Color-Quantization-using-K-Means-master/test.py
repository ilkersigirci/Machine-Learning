#IMPORTS
#from IPython import display
import numpy as np
#import matplotlib.pyplot as plt
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
import six
from io import StringIO

image_name="/mnt/d/DERS/0000GITHUB/Machine-Learning/THE2/Docs/ankara.jpg"



#To load the image as an numpy array
image = misc.imread(image_name)
print(image)
print(image.shape)

