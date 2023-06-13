import numpy as np

def kernel_function(pi, pj):
    r = np.abs(pi - pj)
    phi_r = 1 / (1 + r**2)
    return phi_r

def compute_matrix_A(*args):
    """ args is a positional argument with different number of items """
    size_A = len(args)
    A = np.zeros((size_A, size_A))
    for i in range(size_A):
        for j in range(size_A):
            A[i, j] = kernel_function(args[i], args[j])

    return A


if __name__ == '__main__':
    A = compute_matrix_A(0.3, 0.8, 0.4)
    print(A)
