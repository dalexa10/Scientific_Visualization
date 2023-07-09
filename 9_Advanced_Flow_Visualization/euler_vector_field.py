import numpy as np

def compute_streamline(x0, vec_field, steps, h):
    """
    Computes the streamline for a given vector field
    :param x0: np.array, starting point
    :param vec_field: function, vector field
    :param steps: number of steps to compute streamline
    :param h: stepize
    :return: position of points on streamline
    """
    x = x0
    x_list = []
    for i in range(steps):
        x_list.append(x)
        x = x + h * vec_field(x)
    return np.array(x_list)

if __name__ == '__main__':

    vec_field = lambda x: np.array([2 * x[0] * x[1], (x[0] * x[1])**2])
    x0 = np.array([3, 3])
    h = 0.5
    steps = 3
    x_list = compute_streamline(x0, vec_field, steps, h)
    print('Streamline points are: \n')
    print(x_list)

