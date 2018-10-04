"""Implementation of the DARTS algorithm."""

import numpy as np
from numpy.linalg import lstsq, svd
import sys
import time

# Use the pyfftw interface if it is installed. If not, fall back to the fftpack
# interface provided by SciPy, and finally to numpy if SciPy is not installed.
try:
    import pyfftw.interfaces.numpy_fft as fft
    import pyfftw
    # TODO: Caching and multithreading currently disabled because they give a
    # segfault with dask.
    #pyfftw.interfaces.cache.enable()
    fft_kwargs = {"threads":1, "planner_effort":"FFTW_ESTIMATE"}
except ImportError:
    import scipy.fftpack as fft
    fft_kwargs = {}
except ImportError:
    import numpy.fft as fft
    fft_kwargs = {}

def DARTS(Z, **kwargs):
    """Compute the advection field from a sequence of input images by using the
    DARTS method.

    Parameters
    ----------
    Z : array-like
      Array of shape (T,m,n) containing a sequence of T two-dimensional input
      images of shape (m,n).

    Other Parameters
    ----------------
    N_x : int
      Number of DFT coefficients to use for the input images, x-axis (default=50).
    N_y : int
      Number of DFT coefficients to use for the input images, y-axis (default=50).
    N_t : int
      Number of DFT coefficients to use for the input images, time axis (default=4).
      N_t must be strictly smaller than T.
    M_x : int
      Number of DFT coefficients to compute for the output advection field,
      x-axis  (default=2).
    M_y : int
      Number of DFT coefficients to compute for the output advection field,
      y-axis (default=2).
    print_info : bool
      If True, print information messages.
    lsq_method : {1, 2}
      The method to use for solving the linear equations in the least squares
      sense: 1=numpy.linalg.lstsq, 2=explicit computation of the Moore-Penrose
      pseudoinverse and SVD.
    verbose : bool
        if set to True, it prints information about the program

    Returns
    -------
    out : ndarray
      Three-dimensional array (2,H,W) containing the dense x- and y-components
      of the motion field.

    References
    ----------
    :cite:`RCW2011`

    """
    N_x = kwargs.get("N_x", 50)
    N_y = kwargs.get("N_y", 50)
    N_t = kwargs.get("N_t", 4)
    M_x = kwargs.get("M_x", 2)
    M_y = kwargs.get("M_y", 2)
    print_info = kwargs.get("print_info", False)
    lsq_method = kwargs.get("lsq_method", 2)
    verbose             = kwargs.get("verbose", True)

    if N_t >= Z.shape[0]:
        raise ValueError("N_t = %d >= %d = T, but N_t < T required" % (N_t, Z.shape[0]))

    if verbose:
        print("Computing the motion field with the DARTS method.")
        t0 = time.time()

    Z = np.moveaxis(Z, (0, 1, 2), (2, 0, 1))

    T_x = Z.shape[1]
    T_y = Z.shape[0]
    T_t = Z.shape[2]

    if print_info:
        print("-----")
        print("DARTS")
        print("-----")

        print("  Computing the FFT of the reflectivity fields..."),
        sys.stdout.flush()
        starttime = time.time()

    Z = fft.fftn(Z, **fft_kwargs)

    if print_info:
        print("Done in %.2f seconds." % (time.time() - starttime))

        print("  Constructing the y-vector..."),
        sys.stdout.flush()
        starttime = time.time()

    m = (2*N_x+1)*(2*N_y+1)*(2*N_t+1)
    n = (2*M_x+1)*(2*M_y+1)

    y = np.zeros(m, dtype=complex)

    k_t,k_y,k_x = np.unravel_index(np.arange(m), (2*N_t+1, 2*N_y+1, 2*N_x+1))

    for i in range(m):
        k_x_ = k_x[i] - N_x
        k_y_ = k_y[i] - N_y
        k_t_ = k_t[i] - N_t

        Z_ = Z[k_y_, k_x_, k_t_]

        y[i] = k_t_ * Z_

    if print_info:
        print("Done in %.2f seconds." % (time.time() - starttime))

    A = np.zeros((m, n), dtype=complex)
    B = np.zeros((m, n), dtype=complex)

    if print_info:
        print("  Constructing the H-matrix..."),
        sys.stdout.flush()
        starttime = time.time()

    c1 = -1.0*T_t / (T_x * T_y)

    kp_y,kp_x = np.unravel_index(np.arange(n), (2*M_y+1, 2*M_x+1))

    for i in range(m):
        k_x_  = k_x[i] - N_x
        k_y_  = k_y[i] - N_y
        k_t_  = k_t[i] - N_t

        kp_x_ = kp_x[:] - M_x
        kp_y_ = kp_y[:] - M_y

        i_ = k_y_ - kp_y_
        j_ = k_x_ - kp_x_

        Z_ = Z[i_, j_, k_t_]

        c2 = c1 / T_y * i_
        A[i, :] = c2 * Z_

        c2 = c1 / T_x * j_
        B[i, :] = c2 * Z_

    if print_info:
        print("Done in %.2f seconds." % (time.time() - starttime))

        print("  Solving the linear systems..."),
        sys.stdout.flush()
        starttime = time.time()

    if lsq_method == 1:
        x = lstsq(np.hstack([A, B]), y, rcond=0.01)[0]
    else:
        x = _leastsq(A, B, y)

    if print_info:
        print("Done in %.2f seconds." % (time.time() - starttime))

    h,w = 2*M_y+1,2*M_x+1

    U = np.zeros((h, w), dtype=complex)
    V = np.zeros((h, w), dtype=complex)

    i,j = np.unravel_index(np.arange(h*w), (h, w))

    V[i, j] = x[0:h*w]
    U[i, j] = x[h*w:2*h*w]

    k_x,k_y = np.meshgrid(np.arange(-M_x, M_x+1), np.arange(-M_y, M_y+1))

    U = np.real(fft.ifft2(_fill(U, Z.shape[0], Z.shape[1], k_x, k_y), **fft_kwargs))
    V = np.real(fft.ifft2(_fill(V, Z.shape[0], Z.shape[1], k_x, k_y), **fft_kwargs))

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    # TODO: Sometimes the sign of the advection field is wrong. This appears to
    # depend on N_t...
    return np.stack([U, V])

def _leastsq(A, B, y):
    M = np.hstack([A, B])
    M_ct = M.conjugate().T
    MM = np.dot(M_ct, M)

    M = None

    U,s,V = svd(MM, full_matrices=False)
    MM = None
    mask = s > 0.01*s[0]
    s = 1.0 / s[mask]

    MM_inv = np.dot(np.dot(V[:len(s), :].conjugate().T, np.diag(s)),
                    U[:, :len(s)].conjugate().T)

    return np.dot(MM_inv, np.dot(M_ct, y))

def _fill(X, h, w, k_x, k_y):
    X_f = np.zeros((h, w), dtype=complex)
    X_f[k_y, k_x] = X

    return X_f
