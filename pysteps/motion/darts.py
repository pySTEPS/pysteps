# -*- coding: utf-8 -*-
"""
pysteps.motion.darts
====================

Implementation of the DARTS algorithm.

.. autosummary::
    :toctree: ../generated/

    DARTS
"""

import numpy as np
import time
from numpy.linalg import lstsq, svd

from pysteps import utils
from pysteps.decorators import check_input_frames


@check_input_frames(just_ndim=True)
def DARTS(input_images, **kwargs):
    """Compute the advection field from a sequence of input images by using the
    DARTS method. :cite:`RCW2011`

    Parameters
    ----------
    input_images: array-like
      Array of shape (T,m,n) containing a sequence of T two-dimensional input
      images of shape (m,n).

    Other Parameters
    ----------------
    N_x: int
        Number of DFT coefficients to use for the input images, x-axis (default=50).
    N_y: int
        Number of DFT coefficients to use for the input images, y-axis (default=50).
    N_t: int
        Number of DFT coefficients to use for the input images, time axis (default=4).
        N_t must be strictly smaller than T.
    M_x: int
        Number of DFT coefficients to compute for the output advection field,
        x-axis  (default=2).
    M_y: int
        Number of DFT coefficients to compute for the output advection field,
        y-axis (default=2).
    fft_method: str
        A string defining the FFT method to use, see utils.fft.get_method.
        Defaults to 'numpy'.
    output_type: {"spatial", "spectral"}
        The type of the output: "spatial"=apply the inverse FFT to obtain the
        spatial representation of the advection field, "spectral"=return the
        (truncated) DFT representation.
    n_threads: int
        Number of threads to use for the FFT computation. Applicable if
        fft_method is 'pyfftw'.
    verbose: bool
        If True, print information messages.
    lsq_method: {1, 2}
        The method to use for solving the linear equations in the least squares
        sense: 1=numpy.linalg.lstsq, 2=explicit computation of the Moore-Penrose
        pseudoinverse and SVD.
    verbose: bool
        if set to True, it prints information about the program

    Returns
    -------
    out: ndarray
        Three-dimensional array (2,m,n) containing the dense x- and y-components
        of the motion field in units of pixels / timestep as given by the input
        array R.

    """

    N_x = kwargs.get("N_x", 50)
    N_y = kwargs.get("N_y", 50)
    N_t = kwargs.get("N_t", 4)
    M_x = kwargs.get("M_x", 2)
    M_y = kwargs.get("M_y", 2)
    fft_method = kwargs.get("fft_method", "numpy")
    output_type = kwargs.get("output_type", "spatial")
    lsq_method = kwargs.get("lsq_method", 2)
    verbose = kwargs.get("verbose", True)

    if N_t >= input_images.shape[0] - 1:
        raise ValueError(
            "N_t = %d >= %d = T-1, but N_t < T-1 required"
            % (N_t, input_images.shape[0] - 1)
        )

    if output_type not in ["spatial", "spectral"]:
        raise ValueError(
            "invalid output_type=%s, must be 'spatial' or 'spectral'" % output_type
        )

    if verbose:
        print("Computing the motion field with the DARTS method.")
        t0 = time.time()

    input_images = np.moveaxis(input_images, (0, 1, 2), (2, 0, 1))

    fft = utils.get_method(
        fft_method,
        shape=input_images.shape[:2],
        fftn_shape=input_images.shape,
        **kwargs,
    )

    T_x = input_images.shape[1]
    T_y = input_images.shape[0]
    T_t = input_images.shape[2]

    if verbose:
        print("-----")
        print("DARTS")
        print("-----")

        print("  Computing the FFT of the reflectivity fields...", end="", flush=True)
        starttime = time.time()

    input_images = fft.fftn(input_images)

    if verbose:
        print("Done in %.2f seconds." % (time.time() - starttime))

        print("  Constructing the y-vector...", end="", flush=True)
        starttime = time.time()

    m = (2 * N_x + 1) * (2 * N_y + 1) * (2 * N_t + 1)
    n = (2 * M_x + 1) * (2 * M_y + 1)

    y = np.zeros(m, dtype=complex)

    k_t, k_y, k_x = np.unravel_index(
        np.arange(m), (2 * N_t + 1, 2 * N_y + 1, 2 * N_x + 1)
    )

    for i in range(m):
        k_x_ = k_x[i] - N_x
        k_y_ = k_y[i] - N_y
        k_t_ = k_t[i] - N_t

        y[i] = k_t_ * input_images[k_y_, k_x_, k_t_]

    if verbose:
        print("Done in %.2f seconds." % (time.time() - starttime))

    A = np.zeros((m, n), dtype=complex)
    B = np.zeros((m, n), dtype=complex)

    if verbose:
        print("  Constructing the H-matrix...", end="", flush=True)
        starttime = time.time()

    c1 = -1.0 * T_t / (T_x * T_y)

    kp_y, kp_x = np.unravel_index(np.arange(n), (2 * M_y + 1, 2 * M_x + 1))

    for i in range(m):
        k_x_ = k_x[i] - N_x
        k_y_ = k_y[i] - N_y
        k_t_ = k_t[i] - N_t

        kp_x_ = kp_x[:] - M_x
        kp_y_ = kp_y[:] - M_y

        i_ = k_y_ - kp_y_
        j_ = k_x_ - kp_x_

        R_ = input_images[i_, j_, k_t_]

        c2 = c1 / T_y * i_
        A[i, :] = c2 * R_

        c2 = c1 / T_x * j_
        B[i, :] = c2 * R_

    if verbose:
        print("Done in %.2f seconds." % (time.time() - starttime))

        print("  Solving the linear systems...", end="", flush=True)
        starttime = time.time()

    if lsq_method == 1:
        x = lstsq(np.hstack([A, B]), y, rcond=0.01)[0]
    else:
        x = _leastsq(A, B, y)

    if verbose:
        print("Done in %.2f seconds." % (time.time() - starttime))

    h, w = 2 * M_y + 1, 2 * M_x + 1

    U = np.zeros((h, w), dtype=complex)
    V = np.zeros((h, w), dtype=complex)

    i, j = np.unravel_index(np.arange(h * w), (h, w))

    V[i, j] = x[0 : h * w]
    U[i, j] = x[h * w : 2 * h * w]

    k_x, k_y = np.meshgrid(np.arange(-M_x, M_x + 1), np.arange(-M_y, M_y + 1))

    if output_type == "spatial":
        U = np.real(
            fft.ifft2(_fill(U, input_images.shape[0], input_images.shape[1], k_x, k_y))
        )
        V = np.real(
            fft.ifft2(_fill(V, input_images.shape[0], input_images.shape[1], k_x, k_y))
        )

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    return np.stack([U, V])


def _leastsq(A, B, y):
    M = np.hstack([A, B])
    M_ct = M.conjugate().T
    MM = np.dot(M_ct, M)

    U, s, V = svd(MM, full_matrices=False)

    mask = s > 0.01 * s[0]
    s = 1.0 / s[mask]

    MM_inv = np.dot(
        np.dot(V[: len(s), :].conjugate().T, np.diag(s)), U[:, : len(s)].conjugate().T
    )

    return np.dot(MM_inv, np.dot(M_ct, y))


def _fill(X, h, w, k_x, k_y):
    X_f = np.zeros((h, w), dtype=complex)
    X_f[k_y, k_x] = X

    return X_f
