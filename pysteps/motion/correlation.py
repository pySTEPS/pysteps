# -*- coding: utf-8 -*-
"""
pysteps.motion.correlation
========================

Implementation of the classical correlation based motion vector.

.. autosummary::
    :toctree: ../generated/

    correlation
"""

import numpy as np
import math

from pysteps.decorators import check_input_frames
from pysteps.utils.cleansing import detect_outliers
from pysteps import utils

# To delete
import pdb

# Check if numba can be imported
try:
    from numba import jit

    NUMBA_IMPORTED = True
except ImportError:
    NUMBA_IMPORTED = False


@check_input_frames(2)
def correlation(
    input_images,
    settings={},
    interp_method="idwinterp2d",
    interp_kwargs=None,
    dense=True,
    nr_std_outlier=3,
    k_outlier=30,
    verbose=False,
):
    """
    Run the correlation based optical flow method and interpolate the motion
    vectors. This is using NUMBA to speed up the correlation computation.

    The sparse motion vectors are finally interpolated to return the whole
    motion field.

    Parameters
    ----------
    input_images: ndarray_
        Array of shape (T, m, n) containing a sequence of *T* two-dimensional
        input images of shape (m, n). The indexing order in **input_images** is
        assumed to be (time, latitude, longitude).

        *T* = 2 is the minimum required number of images.
        With *T* > 2, all the resulting obtained vectors are pooled together for
        the final interpolation on a regular grid.

    settings: dict, optional
        Optional dictionary containing keyword arguments for the correlation
        based algorithm.

    interp_method: {"idwinterp2d", "rbfinterp2d"}, optional
      Name of the interpolation method to use. See interpolation methods in
      :py:mod:`pysteps.utils.interpolate`.

    interp_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the interpolation
        algorithm. See the documentation of :py:mod:`pysteps.utils.interpolate`.

    dense: bool, optional
        If True, return the three-dimensional array (2, m, n) containing
        the dense x- and y-components of the motion field.

        If False, return the sparse motion vectors as 2-D **xy** and **uv**
        arrays, where **xy** defines the vector positions, **uv** defines the
        x and y direction components of the vectors.

    nr_std_outlier: int, optional
        Maximum acceptable deviation from the mean in terms of number of
        standard deviations. Any sparse vector with a deviation larger than
        this threshold is flagged as outlier and excluded from the
        interpolation.
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    k_outlier: int or None, optional
        The number of nearest neighbors used to localize the outlier detection.
        If set to None, it employs all the data points (global detection).
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    verbose: bool, optional
        If set to True, print some information about the program.

    Returns
    -------
    out: ndarray_ or tuple
        If **dense=True** (the default), return the advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of the motion
        vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep, where timestep is the
        time difference between the two input images.
        Return a zero motion field of shape (2, m, n) when no motion is
        detected.

        If **dense=False**, it returns a tuple containing the 2-dimensional
        arrays **xy** and **uv**, where x, y define the vector locations,
        u, v define the x and y direction components of the vectors.
        Return two empty arrays when no motion is detected.

    References
    ----------
    Haiden, T., A. Kann, C. Wittmann, G. Pistotnik, B. Bica, and C. Gruber, 2011: The
    Integrated Nowcasting through Comprehensive Analysis (INCA) System and Its
    Validation over the Eastern Alpine Region. Wea. Forecasting, 26, 166–183.
    """

    if not NUMBA_IMPORTED:
        raise ImportError(
            "Numba is not installed. The correlation function cannot be loaded."
        )

    input_images = input_images.copy()

    if interp_kwargs is None:
        interp_kwargs = dict()

    if verbose:
        print("Computing the motion field with the classical correlation-based method.")
        t0 = time.time()

    nr_fields = input_images.shape[0]
    domain_size = (input_images.shape[1], input_images.shape[2])

    PMIN = settings.get("PMIN", 0.05)
    nthin = settings.get("nthin", 15)

    NI = domain_size[0]
    NJ = domain_size[1]

    NX = int(np.ceil(NI / nthin))
    NY = int(np.ceil(NJ / nthin))

    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    X, Y = xgrid[::nthin], ygrid[::nthin]
    XX, YY = np.meshgrid(X, Y, indexing="ij")

    U_t = []
    V_t = []

    for n in range(nr_fields - 1):
        # extract consecutive images
        prvs_img = input_images[n, :, :].copy()
        next_img = input_images[n + 1, :, :].copy()

        # Removing values below the PMIN threshold
        prvs_img[prvs_img < PMIN] = 0.0
        next_img[next_img < PMIN] = 0.0

        Uana = np.full((NJ, NI), np.nan)[::nthin, ::nthin]
        Vana = np.full((NJ, NI), np.nan)[::nthin, ::nthin]

        Uana, Vana = compute_motion_numba(next_img, prvs_img, NI, NJ, nthin, Uana, Vana)

        U_t.append(Uana)
        V_t.append(Vana)

    Uana = np.nanmean(np.array(U_t), axis=0)
    Vana = np.nanmean(np.array(V_t), axis=0)

    ## Choose computed boxes
    boole_ = np.isfinite(Uana) & np.isfinite(Vana)

    if np.sum(boole_) == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    uv = np.vstack([Uana[boole_], Vana[boole_]]).T
    xy = np.vstack([XX[boole_], YY[boole_]]).T

    # detect and remove outliers
    outliers = detect_outliers(uv, nr_std_outlier, xy, k_outlier, verbose)
    xy = xy[~outliers, :]
    uv = uv[~outliers, :]

    if verbose:
        print("--- LK found %i sparse vectors ---" % xy.shape[0])

    # return sparse vectors if required
    if not dense:
        return xy, uv

    # # interpolation
    interpolation_method = utils.get_method(interp_method)
    uvgrid = interpolation_method(xy, uv, xgrid, ygrid, **interp_kwargs)

    if verbose:
        print("--- total time: %.2f seconds ---" % (time.time() - t0))

    return uvgrid


if NUMBA_IMPORTED:

    @jit(nopython=True)
    def compute_motion_numba(image1, image2, NI, NJ, nthin, IS_tmp, JS_tmp):
        """
        Compute the motion between two images using a correlation-based method optimized with Numba.

        Parameters:
        -----------
        image1 : 2D numpy array
            The first input image for motion computation.
        image2 : 2D numpy array
            The second input image for motion computation.
        NI : int
            The number of rows in the input images.
        NJ : int
            The number of columns in the input images.
        nthin : int
            The thinning factor for the grid used in motion computation.
        IS_tmp : 2D numpy array
            The output array to store the computed motion in the x-direction.
        JS_tmp : 2D numpy array
            The output array to store the computed motion in the y-direction.

        Returns:
        --------
        IS_tmp : 2D numpy array
            The updated motion array in the x-direction after computation.
        JS_tmp : 2D numpy array
            The updated motion array in the y-direction after computation.

        Notes:
        ------
        - This function computes the motion by correlating patches of the first image (`image1`)
          with patches of the second image (`image2`).
        - The computation is performed on a grid defined by the `nthin` parameter to reduce
          computational complexity.
        - The function uses a fixed-size search window (`nsh`) and a quantization parameter (`nqu`)
          to define the neighborhood for correlation computation.
        - The correlation is calculated within a defined window and the maximum correlation value is
          used to determine the motion vector.
        - The function utilizes several optimization techniques to ensure efficient computation
          and is compiled with Numba for further speed-up.

        """

        ## List of parameters from the original paper (TODO: Optimization as input values in func.)
        nsh = 20
        nqu = 45
        nsq = nsh + nqu
        di = 1
        PAREAMIN = 1.0
        rr_dpa_min = 0.03
        nn = (2 * nqu / di + 1) ** 2.0

        for i in range(0, NI, nthin):
            if (i >= nsq) and (i < NI - nsq):
                for j in range(0, NJ, nthin):
                    if (j >= nsq) and (j < NJ - nsq):
                        ii1 = max(i - nqu, 0)
                        ii2 = min(i + nqu, NI - 1)
                        jj1 = max(j - nqu, 0)
                        jj2 = min(j + nqu, NJ - 1)

                        sy = 0.0
                        sy2 = 0.0

                        for ii in range(ii1, ii2 + 1, di):
                            for jj in range(jj1, jj2 + 1, di):
                                sy = sy + image1[jj, ii]
                                sy2 = sy2 + image1[jj, ii] ** 2.0

                        sigy = sy2 - sy**2.0 / nn
                        isho = -99
                        jsho = -99

                        if (sigy > 0.0) and (sy > PAREAMIN):
                            corqx = 0.1
                            for ish in range(-nsh, nsh + 1):
                                for jsh in range(-nsh, nsh + 1):
                                    if math.sqrt((ish) ** 2.0 + (jsh) ** 2.0) > nsh:
                                        continue
                                    sx = 0.0
                                    sx2 = 0.0
                                    sxy = 0.0
                                    for ii in range(ii1, ii2 + 1, di):
                                        for jj in range(jj1, jj2 + 1, di):
                                            ind_x = min(max(0, jj + jsh), NJ - 1)
                                            ind_y = min(max(0, ii + ish), NI - 1)
                                            sx += image2[ind_x, ind_y]
                                            sx2 += image2[ind_x, ind_y] ** 2.0
                                            sxy += image2[ind_x, ind_y] * image1[jj, ii]

                                    sigx = sx2 - sx**2.0 / nn
                                    if sigx > 0.0:
                                        corq = (sxy - sx * sy / nn) ** 2.0 / (
                                            sigx * sigy
                                        )
                                        if corq > corqx:
                                            corqx = corq
                                            isho = ish
                                            jsho = jsh

                            if (
                                (isho != -99)
                                and (corqx > 0.3)
                                and (corqx * sy > 1.0)
                                and (sigy / sy >= 0.08)
                            ):
                                if (abs(isho) < nsh) and (abs(jsho) < nsh):
                                    IS_tmp[int(j / nthin), int(i / nthin)] = -isho
                                    JS_tmp[int(j / nthin), int(i / nthin)] = -jsho
                            else:
                                IS_tmp[int(j / nthin), int(i / nthin)] = 0
                                JS_tmp[int(j / nthin), int(i / nthin)] = 0

        return IS_tmp, JS_tmp

else:

    def compute_motion_numba(image1, image2, NI, NJ, nthin, IS_tmp, JS_tmp):
        raise RuntimeError(
            "Numba is not available, so the compute_motion_numba function cannot be used."
        )
