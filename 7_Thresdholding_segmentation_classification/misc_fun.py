import numpy as np
import math

def compute_affinity_IG_FG(lamb, P_F):
    """
    Computes the affinity of non-scribble pixel for the foreground for binary segmentation
    Inputs:
        :param lamb: (float) given parameter set by user (check notes)
        :param P_F: (float) probability Foreground
        :return: aff_FG (float) affinity foreground
    """
    P_B = 1 - P_F  # Probability foreground
    aff_FG = - lamb * np.log(P_B)
    return aff_FG

def compute_affinity_GC(p1, p2, sigma, metric='distance'):
    """

    :param p1: (list) [0]: np.array(x, y) position pixel 1   [1]: float, function_value for pixel 1
    :param p2: (list) [0]: np.array(x, y) position pixel 2   [1]: float, function_value for pixel 2
    :param sigma: (float) variance between the two distributions
    :param metric: (str) default = 'distance'  -> Euclidean distance. Other value can be 'intensity'
    :return: affinity (float)
    """
    if metric == 'distance':
        num = (math.dist(p1[0], p2[0]))**2
        aff = np.exp(- num / (2 * sigma**2))
        return aff
    elif metric == 'intensity':
        num = np.abs(p1[1] - p2[1])
        aff = np.exp(- num / (2 * sigma**2))
        return aff
    else:
        raise TypeError('Check provided arguments')


if __name__ == '__main__':

    # --------------------------------------------
    #                  Affinity
    # --------------------------------------------
    # Test Affinity
    np.testing.assert_allclose(compute_affinity_IG_FG(lamb=0.9, P_F=0.7),  1.08357552389, rtol=1e-2)

    # Example with lambda = 0.7 and P_F = 0.1
    aff = compute_affinity_IG_FG(0.7, 0.1)
    print(aff)

    # --------------------------------------------
    #                Graph Cut
    # --------------------------------------------
    p1 = [np.array([1, 1]), 3]
    p2 = [np.array([0, 1]), 4]

    aff_dist = compute_affinity_GC(p1, p2, sigma=2)
    print('Affinity using Euclidean distance is {:.4f}'.format(aff_dist))

    aff_inten = compute_affinity_GC(p1, p2, sigma=2, metric='intensity')
    print('Affinity using Euclidean distance is {:.4f}'.format(aff_inten))