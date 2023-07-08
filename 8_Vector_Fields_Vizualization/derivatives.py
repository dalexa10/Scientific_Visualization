import cmath
import numpy as np
from sympy import symbols, diff

def compute_numerical_derivative_FD(f, x, dx=1e-6):
    """
    Computes the numerical derivative of a function f at a point x
    :param f: function, f(x)
    :param x: point at which derivative is to be computed
    :param dx: delta x
    :return: df/dx
    """
    df = (f(x + dx) - f(x - dx)) / (2 * dx)
    return df

def comppute_numerical_derivative_CS(f, x, h=1e-6):
    """
    Computes the numerical derivative of a function f at a point x with complex step method
    :param f: function, f(x) NOTE: f should work with complex numbers, check cmath library
    :param x: point at which derivative is to be computed
    :param h: complex step size
    :return: df/dx
    """
    df = f(complex(x, h)).imag / h
    return df

def compute_symbolic_divergence(vector_field):
    """
    Computes the symbolic divergence of a vector field
    :param vector_field: np.array (Nx3), N: number of points, 3: x, y, z
    :return: divergence
    """
    x, y, z = symbols('x y z')
    f = vector_field[0]
    g = vector_field[1]
    h = vector_field[2]
    div = diff(f, x) + diff(g, y) + diff(h, z)
    return div


def compute_symbolic_curl(vector_field):
    """
    Computes the symbolic curl of a vector field
    :param vector_field: np.array (Nx3), N: number of points, 3: x, y, z
    :return: curl
    """
    x, y, z = symbols('x y z')
    f = vector_field[0]
    g = vector_field[1]
    h = vector_field[2]
    curl = [diff(h, y) - diff(g, z),
            diff(f, z) - diff(h, x),
            diff(g, x) - diff(f, y)]
    return curl


if __name__ == '__main__':

    # Parabola example
    def f_parabola(a, b, c):
        f = lambda x: a * x**2 + b * x + c
        return f
    a, b, c = [1, 2, 3]
    x = 2
    f1_ex = f_parabola(a, b, c)
    df1_dx_FD = compute_numerical_derivative_FD(f1_ex, x)
    df1_dx_CS = comppute_numerical_derivative_CS(f1_ex, x)

    print('Derivative at x = {:.3f} is {:.3f} '.format(x, df1_dx_FD))
    print('Derivative at x = {:.3f} is {:.3f} '.format(x, df1_dx_CS))

    # Another example
    f2 = lambda x: np.sin(x) * np.log(x**2)
    df2_dx_FD = compute_numerical_derivative_FD(f2, x)
    df2_dx_CS = comppute_numerical_derivative_CS(f2, x)

    print('Derivative at x = {:.3f} is {:.3f} '.format(x, df2_dx_FD))
    print('Derivative at x = {:.3f} is {:.3f} '.format(x, df2_dx_CS))


    # Vector field divergence example
    x, y, z = symbols('x y z')
    vf = [x**2 * z**3, x**2 * y**2 * z, x * y**3 * z]
    div = compute_symbolic_divergence(vf)
    print('Symbolic divergence is: ', div)

    # If you were to get the numerical value, just do this
    X_eval = [5, 2, 5]
    print('Numerical divergence at X= {}  is {}'.format(X_eval, div.subs([(x, X_eval[0]), (y, X_eval[1]), (z, X_eval[2])])))


    # Vector field curl example
    x, y, z = symbols('x y z')
    vf_2 = [x**2 * y * z**2, x * y**3 * z, y**2 * z**2]
    curl = compute_symbolic_curl(vf_2)
    print('Symbolic curl is: ', curl)

    # If you were to get the numerical value, just do this
    X_eval_2 = [5, 3, 7]
    curl_num = [i.subs(x, X_eval_2[0]).subs(y, X_eval_2[1]).subs(z, X_eval_2[2]) for i in curl]
    print('Numerical curl at X= {}  is {}'.format(X_eval, curl_num))
