import numpy as np
from numpy import linalg

def compute_directional_gradient(f1, f2, x1, x2):
    df_dx = (f2 - f1) / (x2 - x1)
    return df_dx

def compute_normalized_normal_vector(f_vec, x_vec):
    """
    Computes the normalized normal vector by computing the gradient in all the given directions
    with central difference finite difference method
    :param f_vec: np.array (Nx2), N: directions 2: Columns for f(x + h, y, z), f(x - h, y, z)
    :param x_vec: np.array(Nx2)  N: directions 2: Corresponding coordinates (x+h, x-h)
    :return:
    """
    grad = []
    for i in range(f_vec.shape[0]):
        df_dx_i = compute_directional_gradient(f_vec[i][0], f_vec[i][1], x_vec[i][0], x_vec[i][1])
        grad.append(df_dx_i)

    grad_array = np.array(grad)
    grad_norm_vec = grad_array / linalg.norm(grad_array)
    return grad_norm_vec


if __name__ == '__main__':

    # f = np.array([[f1, f2],
    #               [g1, g2],
    #               .....])

    # Note: coordinates and functions should match accordingly

    f_vec = np.array([[1, 0.8],
                      [0.2, 0.6],
                      [0.5, 0.3]])
    x_vec = np.array([[5, 7],
                      [2, 4],
                      [1, 3]])
    df_dx = compute_normalized_normal_vector(f_vec, x_vec)
    print('Gradient vector is : \n')
    print(df_dx)