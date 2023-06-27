import numpy as np

def over_operator(cf, alpha_f, cb, alpha_b):
    """ Computes the RGB value of two compositing colors given
        cf: np.array (3x1) RGB value of front color
        alpha_f: float, front color opacity
        cb:  np.array (3x1) RGB value of back color
        alpha_b: float, back color opacity
    """

    c = alpha_f * cf + (1 - alpha_f) * alpha_b * cb
    alpha = alpha_f + (1 - alpha_f) * alpha_b
    return c, alpha


def compute_compositing_color(color_ls):
    """ Given a color list as following below, iterates over the list and computes the compositing color
        color_ls (list):
            example: color_ls = [[c_1, alpha_1],    # Front
                                 [c_2, alpha_2],    # Mid close to front
                                 [c_3, alpha_3],    # Mid close to back
                                 [c_4, alpha_4]]    # Back
    """
    c_f, alpha_f = color_ls[-2][0], color_ls[-2][1]
    c_b, alpha_b = color_ls[-1][0], color_ls[-1][1]

    c_new, alpha_new = over_operator(c_f, alpha_f, c_b, alpha_b)

    if len(color_ls) >= 3:
        for i in range(len(color_ls) - 2, 0, -1):
            # Note: alpha_b = 1. because already premultiplied. Check notes for more info
            c_new, alpha_new = over_operator(cf=color_ls[i][0], alpha_f=color_ls[i][1],
                                             cb=c_new, alpha_b=1.)

    return c_new, alpha_new

if __name__ == '__main__':

    # Example 1
    color_ls = [[np.array([[0, 1, 1]]).T, 0.4],
                [np.array([[0, 1, 0]]).T, 0.4],
                [np.array([[1, 0, 0]]).T, 0.9]]
    c_new, alpha_new = compute_compositing_color(color_ls)
    print('Compositing color Example 1')
    print(c_new, '\n')
    print(alpha_new, '\n')

    # Example 2
    color_ls_2 = [[np.array([[0.8, 0.4, 0.3]]).T, 0.3],
                  [np.array([[0.9, 0.2, 0.2]]).T, 1.]]
    c_new_2, alpha_new_2 = compute_compositing_color(color_ls_2)
    print('Compositing color Example 2')
    print(c_new_2, '\n')
    print(alpha_new_2, '\n')





