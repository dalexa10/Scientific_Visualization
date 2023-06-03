__author__ = 'Dario Rodriguez'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap


tf_dict = {'red': [[0.0,  0.0, 0.0],
                   [0.2,  0.0, 0.14],
                   [0.75, 1.0, 1.0],
                   [1.0,  0.7, 0.7]],
          'green': [[0.0,  0.0, 0.14],
                    [0.24, 0.14, 0.24],
                    [0.3, 0.7, 0.7],
                    [0.4, 0.3, 0.0],
                    [1.0, 0.0, 0.0]],
          'blue':  [[0.0,  1.0, 1.0],
                    [0.28,  0.0, 0.0],
                    [1.0,  0.0, 0.0]]}

def gray2lum(im_array):
    """
    Convert gray RGB image to single value pixel luminance image
    Simple slicing stuff
    """
    im_lum = im_array[:, :, 0]
    return im_lum


def create_mars_cmap(tf_dict):
    """
    Define a tailored linear segmented colormap given the transfer
    functions in a dict-wise
    """
    mars_cmap = LinearSegmentedColormap('mars_cmap', segmentdata=tf_dict, N=256)
    return mars_cmap


mars_gray = mpimg.imread('mars.png')
mars_lum = gray2lum(mars_gray)
mars_colormap = create_mars_cmap(tf_dict)

mars_red = mars_colormap(mars_lum)

fig, ax = plt.subplots()
ax.imshow(mars_red)

plt.show()