import numpy as np

def compute_barycentric_coord(t_cd, p_cd):
    """
    Computes the parameters b1, b2, b3 (barycentric coordinates) as fuction of
    :param t_cd: np.array((3, 2)) --> Triangle vertices in (x, y) coordinates
    :param p_cd: np.array((2, 1)) --> (x, y) coordinates of p point
    :return:
        np.array([b1, b2, b3]) --> Barycentric coordinates
    """
    x1, x2, x3 = [t_cd[i][0] for i in range(t_cd.shape[0])]
    y1, y2, y3 = [t_cd[i][1] for i in range(t_cd.shape[0])]
    px, py = p_cd[0], p_cd[1]

    b1 = ((py - y3) * (x2 - x3) + (y2 - y3) * (x3 - px)) / ((y1 - y3) * (x2 - x3) + (y2 - y3) * (x3 - x1))
    b2 = ((py - y1) * (x3 - x1) + (y3 - y1) * (x1 - px)) / ((y1 - y3) * (x2 - x3) + (y2 - y3) * (x3 - x1))
    b3 = ((py - y2) * (x1 - x2) + (y1 - y2) * (x2 - px)) / ((y1 - y3) * (x2 - x3) + (y2 - y3) * (x3 - x1))

    return np.array([b1, b2, b3])

def compute_interpolation_barycentric(b_array, f_array):
    """
    Computes the interpolation given a triangle and a set of barycentric coordinates
    :param b_array(3, 1):
    :param f_array(3, 1):
    :return:
        f(p): float, interpolated function at p (based on barycentric coordinates)
    """
    f_p = np.sum(b_array * f_array)
    return f_p




if __name__ == '__main__':
    t_vert = np.array([[0, -3],
                   [0, 3],
                   [6, 0]])
    f_t_vert = np.array([[4, 9, 20],    # [[R0, G0, B0],
                         [24, 0, 18],   #  [R1, G1, B1],
                         [12, 3, 15]])  #  [R2, G2, B2]]
    p_cd = np.array([2, 2])



    b_array = compute_barycentric_coord(t_vert, p_cd)

    print('Barycentric coordiantes are: \n'
          'b1 = {:.3f} \n'
          'b2 = {:.3f} \n'
          'b3 = {:.3f}'.format(b_array[0], b_array[1], b_array[2]))


    print('Interpolating for RGB values in triangle vertices')
    RGB_p = np.empty(3, dtype=float)

    for i in range(3):
        f_array = f_t_vert[:, i]
        f_p = compute_interpolation_barycentric(b_array, f_array)
        RGB_p[i] = f_p

    print('Interpolated RGB values at given p point are: \n'
          'R = {:.3f} \n'
          'G = {:.3f} \n'
          'B = {:.3f}'.format(RGB_p[0], RGB_p[1], RGB_p[2]))

