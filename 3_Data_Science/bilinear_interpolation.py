import numpy as np

def linear_interpolation(x1, f1, x2, f2, x):
    t = (x - x1) / (x2 - x1)
    f_int = (1 - t) * f1 + t * f2
    return f_int

def compute_bilinear_interpolation(pt_c_array, f_array, x_int):
    """
    Given a set of 4 points with their value functions at each one, computes a bilinear interpolation
    at x_int coordinates
    Note: Points should be provided in a counter-clockwise order based on their plane position
    :param pt_cd_array: np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    :param f_array: np.array([f0, f1, f2, f3])
    :param x_int: np.array([x, y])
    :return:
        f_int (float): Interpoalted value at x_int
    """
    # x_axis interpolation first
    x0, x1, x3, x2 = [pt_c_array[i][0] for i in range(4)]
    y0, y1, y3, y2 = [pt_c_array[i][1] for i in range(4)]
    f0, f1, f3, f2 = f_array
    x, y = x_int

    # Interpolation in x axis
    # Verfiy coordinates
    if x0 == x2 and x1 == x3:
        f_a = linear_interpolation(x0, f0, x1, f1, x)
        f_b = linear_interpolation(x2, f2, x3, f3, x)
    else:
        raise ValueError('Coordinates not provided in anti-clockwise order')

    # Interpolation in y axis
    # Verify coordiantes
    if y0 == y1 and y2 == y3:
        f = linear_interpolation(y0, f_a, y2, f_b, y)
        return f
    else:
        raise ValueError('Coordinates not provided in anti-clockwise order')


if __name__ == '__main__':
    pt_cd_array = np.array([[8, 9],
                            [9, 9],
                            [9, 10],
                            [8, 10]])
    f_array = np.array([0, 77, 0, 75])
    x_int = np.array([8.5, 9.3])

    f_int = compute_bilinear_interpolation(pt_cd_array, f_array, x_int)
    print('Bilienar interpolated value is {:.3f}'.format(f_int))