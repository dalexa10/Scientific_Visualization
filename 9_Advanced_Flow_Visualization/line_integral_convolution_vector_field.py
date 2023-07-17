__author__ = "Dario Rodriguez"

import numpy as np
import math
import matplotlib.pyplot as plt

def advance(ux, uy, x, y, fx, fy, nx, ny):
    """
    Move to the next pixel in the direction of the vector at the current pixel.

    Parameters
    ----------
    ux : float
      Vector x component.
    uy :float
      Vector y component.
    x : int
      Pixel x index.
    y : int
      Pixel y index.
    fx : float
      Position along x in the pixel unit square.
    fy : float
      Position along y in the pixel unit square.
    nx : int
      Number of pixels along x.
    ny : int
      Number of pixels along y.

    Returns
    -------
    x : int
      Updated pixel x index.
    y : int
      Updated pixel y index.
    fx : float
      Updated position along x in the pixel unit square.
    fy : float
      Updated position along y in the pixel unit square.
    """
    # ------------------------------------------------
    #               Time calculation
    # ------------------------------------------------
    if ux == 0 and uy == 0:
        return None
    else:
        # X axis
        # Ensure tx is not necessarily zero when sudden change of vector direction from pixel to pixel
        if ux < 0.:
            tx = abs(((x + fx) - math.ceil(x + fx) + 1) / ux)
        elif ux > 0.:
            tx = abs((math.floor(x + fx) + 1 - (x + fx)) / ux)
        elif ux == 0.:
            tx = np.inf
        # Y axis
        # Ensure ty is not necessarily zero when sudden change of vector direction from pixel to pixel
        if uy < 0.:
            ty = abs(((y + fy) - math.ceil(y + fy) + 1) / uy)
        elif uy > 0.:
            ty = abs((math.floor(y + fy) + 1 - (y + fy)) / uy)
        elif uy == 0.:
            ty = np.inf

        # ----------------------------------------------
        #            Minimum time check
        # ----------------------------------------------
        if tx < ty:
            if ux < 0:
                fx = 1.
                x -= 1
                if x <= 0:
                    x = 0
            else:
                fx = 0.
                x += 1
                if x >= nx - 1:
                    x = nx - 1
            fy = fy + (uy * tx)
        elif tx > ty:
            if uy < 0:
                fy = 1.
                y -= 1
                if y <= 0:
                    y = 0
            else:
                fy = 0.
                y += 1
                if y >= ny - 1:
                    y = ny - 1
            fx = fx + (ux * ty)
        else:
            if ux > 0 and uy > 0:
                fx = 0
                fy = 0
                x += 1
                y += 1
                if x >= (nx - 1):
                    x = nx - 1
                if y >= (ny - 1):
                    y = ny - 1
            elif ux < 0 and uy > 0:
                fx = 1
                fy = 0
                x -= 1
                y += 1
                if x <= 0:
                    x = 0
                if y >= (ny - 1):
                    y = ny - 1
            elif ux > 0 and uy < 0:
                fx = 0
                fy = 1
                x += 1
                y -= 1
                if x >= nx - 1:
                    x = nx - 1
                if y <= 0:
                    y = 0
            else:
                fx = 1
                fy = 1
                x -= 1
                y -= 1
                if x <= 0:
                    x = 0
                if y <= 0:
                    y = 0
        return (x, y, fx, fy)

def compute_streamline(vx, vy, texture, px, py, kernel):
    """
    Return the convolution of the streamline for the given pixel (px, py).

    Parameters
    ----------
    vx : array (ny, nx)
      Vector field x component.
    vy : array (ny, nx)
      Vector field y component.
    texture : array (ny,nx)
      The input texture image that will be distorted by the vector field.
    px : int
      Pixel x index.
    py : int
      Pixel y index.
    kernel : 1D array
      The convolution kernel: an array weighting the texture along
      the stream line. The kernel should be
      symmetric.
    fx : float
      Position along x in the pixel unit square.
    fy : float
      Position along y in the pixel unit square.
    nx : int
      Number of pixels along x.
    ny : int
      Number of pixels along y.

    Returns
    -------
    sum : float
      Weighted sum of values at each pixel along the streamline that starts at center of pixel (px,py)

    """
    # Initialize fx and fy
    fx_i, fy_i = 0.5, 0.5
    px_i, py_i = px, py
    L = int((kernel.size - 1) / 2)
    Pix_ls = []
    Pix_ls.append([py_i, px_i])
    sum = texture[py_i, px_i] * kernel[L]

    # Going forward from (px, py)
    for i in range(1, L + 1):
        try:
            px_i, py_i, fx_i, fy_i = advance(vx[py_i, px_i], vy[py_i, px_i], px_i, py_i, fx_i, fy_i,
                                             texture.shape[1], texture.shape[0])
            Pix_ls.append([py_i, px_i])
            sum += texture[py_i, px_i] * kernel[L + i]
        except:
            break

    # Going backward from (px, py)
    px_i, py_i = px, py
    fx_i, fy_i = 0.5, 0.5
    for i in range(L - 1):
        try:
            px_i, py_i, fx_i, fy_i = advance(- vx[py_i, px_i], - vy[py_i, px_i], px_i, py_i, fx_i, fy_i,
                                             texture.shape[1], texture.shape[0])
            Pix_ls = [[py_i, px_i]] + Pix_ls
            sum += texture[py_i, px_i] * kernel[L - i - 1]
        except:
            break
    return sum, Pix_ls

def lic(vx, vy, texture, kernel):
    """
    Return an image of the texture array blurred along the local vector field orientation.

    Parameters
    ----------
    vx : array (ny, nx)
      Vector field x component.
    vy : array (ny, nx)
      Vector field y component.
    texture : array (ny,nx)
      The input texture image that will be distorted by the vector field.
    kernel : 1D array
      The convolution kernel: an array weighting the texture along
      the stream line. The kernel should be
      symmetric.

    Returns
    -------
    result : array(ny,nx)
      An image of the texture convoluted along the vector field
      streamlines.

    """
    result = np.zeros((texture.shape[0], texture.shape[1]))
    for j in range(texture.shape[0]):
        for i in range(texture.shape[1]):
            sum, _ = compute_streamline(vx, vy, texture, j, i, kernel)
            result[j, i] = sum / kernel.sum()

    return result


if __name__ == '__main__':

    size = 300

    vortex_spacing = 0.5
    extra_factor = 2.

    a = np.array([1, 0]) * vortex_spacing
    b = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]) * vortex_spacing
    rnv = int(2 * extra_factor / vortex_spacing)
    vortices = [n * a + m * b for n in range(-rnv, rnv) for m in range(-rnv, rnv)]
    vortices = [(x, y) for (x, y) in vortices if -extra_factor < x < extra_factor and -extra_factor < y < extra_factor]

    xs = np.linspace(-1, 1, size).astype(np.float64)[None, :]
    ys = np.linspace(-1, 1, size).astype(np.float64)[:, None]

    vx = np.zeros((size, size), dtype=np.float64)
    vy = np.zeros((size, size), dtype=np.float64)
    for (x, y) in vortices:
        rsq = (xs - x) ** 2 + (ys - y) ** 2
        vx += (ys - y) / rsq
        vy += -(xs - x) / rsq

    size = 300

    np.random.seed(123)
    texture = np.random.rand(size, size).astype(np.float64)
    fig, ax = plt.subplots()
    ax.imshow(texture, cmap="gray")
    plt.show()

    xs = np.linspace(-1, 1, size).astype(np.float64)[None, :]
    ys = np.linspace(-1, 1, size).astype(np.float64)[:, None]

    vx = np.zeros((size, size), dtype=np.float64)
    vy = np.zeros((size, size), dtype=np.float64)
    for (x, y) in vortices:
        rsq = (xs - x) ** 2 + (ys - y) ** 2
        vx += (ys - y) / rsq
        vy += -(xs - x) / rsq

    L = 10  # Radius of the kernel
    kernel = np.sin(np.arange(2 * L + 1) * np.pi / (2 * L + 1)).astype(np.float64)

    image = lic(vx, vy, texture, kernel)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='viridis')
    plt.show()
