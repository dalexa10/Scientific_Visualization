import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import matplotlib.image as mpimg
import copy

def red2green(col_arr, tol=0.95):
    """ TUrns red to green colors for a given color array RGBA and tolerance"""
    if col_arr[0] >= 1 * tol and col_arr[1] <= 1 * (1 - tol) and col_arr[2] <= 1 * (1 - tol):
        col_arr = [0., 1., 0., 1.]
    return col_arr


# Import Mario's image
mario = mpimg.imread("mario_big.png")
luigi = copy.deepcopy(mario)

for i in range(mario.shape[0]):
    for j in range(mario.shape[1]):
        luigi[i, j] = red2green(mario[i, j])

mpimg.imsave("luigi_big.png", luigi)
plt.imshow(luigi)
plt.show()



