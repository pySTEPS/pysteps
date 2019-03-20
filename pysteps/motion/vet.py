# -*- coding: utf-8 -*-
"""
pysteps.motion.vet
==================

Variational Echo Tracking (VET) Module

This module implements the VET algorithm presented
by `Laroche and Zawadzki (1995)`_ and used in the
McGill Algorithm for Prediction by Lagrangian Extrapolation (MAPLE) described
in `Germann and Zawadzki (2002)`_.


.. _`Laroche and Zawadzki (1995)`:\
    http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

.. _`Germann and Zawadzki (2002)`:\
    http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2

The morphing and the cost functions are implemented in Cython and parallelized
for performance.

.. currentmodule:: pysteps.motion.vet

.. autosummary::
    :toctree: ../generated/

    vet
    vet_cost_function
    vet_cost_function_gradient
    morph
    round_int
    ceil_int
    get_padding
"""

import numpy
from numpy.ma.core import MaskedArray
from scipy.ndimage.interpolation import zoom
from scipy.optimize import minimize

from pysteps.motion._vet import _warp, _cost_function


def round_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(numpy.round(scalar))


def ceil_int(scalar):
    """
    Round number to nearest integer. Returns and integer value.
    """
    return int(numpy.ceil(scalar))


def get_padding(dimension_size, sectors):
    """
    Get the padding at each side of the one dimensions of the image
    so the new image dimensions are divided evenly in the
    number of *sectors* specified.

    Parameters
    ----------

    dimension_size : int
        Actual dimension size.

    sectors : int
        number of sectors over which the the image will be divided.

    Return
    ------

    pad_before , pad_after: int, int
        Padding at each side of the image for the corresponding dimension.
    """
    reminder = dimension_size % sectors

    if reminder != 0:
        pad = sectors - reminder
        pad_before = pad // 2
        if pad % 2 == 0:
            pad_after = pad_before
        else:
            pad_after = pad_before + 1

        return pad_before, pad_after

    return 0, 0


def morph(image, displacement, gradient=False):
    """
    Morph image by applying a displacement field (Warping).

    The new image is created by selecting for each position the values of the
    input image at the positions given by the x and y displacements.
    The routine works in a backward sense.
    The displacement vectors have to refer to their destination.

    For more information in Morphing functions see Section 3 in
    `Beezley and Mandel (2008)`_.

    Beezley, J. D., & Mandel, J. (2008).
    Morphing ensemble Kalman filters. Tellus A, 60(1), 131-140.

    .. _`Beezley and Mandel (2008)`: http://dx.doi.org/10.1111/\
    j.1600-0870.2007.00275.x


    The displacement field in x and y directions and the image must have the
    same dimensions.

    The morphing is executed in parallel over x axis.

    The value of displaced pixels that fall outside the limits takes the
    value of the nearest edge. Those pixels are indicated by values greater
    than 1 in the output mask.

    Parameters
    ----------

    image : ndarray (ndim = 2)
        Image to morph

    displacement : ndarray (ndim = 3)
        Displacement field to be applied (Warping). The first dimension
        corresponds to the coordinate to displace.

        The dimensions are: displacement [ i/x (0) or j/y (1) ,
        i index of pixel, j index of pixel ]


    gradient : bool, optional
        If True, the gradient of the morphing function is returned.


    Returns
    -------

    image : ndarray (float64 ,ndim = 2)
        Morphed image.

    mask : ndarray (int8 ,ndim = 2)
        Invalid values mask. Points outside the boundaries are masked.
        Values greater than 1, indicate masked values.

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.

    """

    if not isinstance(image, MaskedArray):
        _mask = numpy.zeros_like(image, dtype='int8')
    else:
        _mask = numpy.asarray(numpy.ma.getmaskarray(image),
                              dtype='int8')

    _image = numpy.asarray(image, dtype='float64', order='C')
    _displacement = numpy.asarray(displacement, dtype='float64', order='C')

    return _warp(_image, _mask, _displacement, gradient=gradient)


def vet_cost_function_gradient(*args, **kwargs):
    """Compute the vet cost function gradient.
    See :py:func:`vet_cost_function` for more information.
    """
    kwargs["gradient"] = True
    return vet_cost_function(*args, **kwargs)


def vet_cost_function(sector_displacement_1d,
                      input_images,
                      blocks_shape,
                      mask,
                      smooth_gain,
                      debug=False,
                      gradient=False):
    """
    .. _`scipy minimization`: \
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html
    
    Variational Echo Tracking Cost Function.

    This function is designed to be used with the `scipy minimization`_.
    The function first argument is the variable to be used in the
    minimization procedure.

    The sector displacement must be a flat array compatible with the
    dimensions of the input image and sectors shape (see parameters section
    below for more details).



    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html


    Parameters
    ----------

    sector_displacement_1d : ndarray_
        Array of displacements to apply to each sector. The dimensions are:
        sector_displacement_2d
        [ x (0) or y (1) displacement, i index of sector, j index of sector ].
        The shape of the sector displacements must be compatible with the
        input image and the block shape.
        The shape should be (2, mx, my) where mx and my are the numbers of
        sectors in the x and the y dimension.

    input_images : ndarray_
        Input images, sequence of 2D arrays, or 3D arrays.
        The first dimension represents the images time dimension.

        The template_image (first element in first dimensions) denotes the
        reference image used to obtain the displacement (2D array).
        The second is the target image.

        The expected dimensions are (2,nx,ny).
        Be aware the the 2D images dimensions correspond to (lon,lat) or (x,y).

    blocks_shape : ndarray_ (ndim=2)
        Number of sectors in each dimension (x and y).
        blocks_shape.shape = (mx,my)

    mask : ndarray_ (ndim=2)
        Data mask. If is True, the data is marked as not valid and is not
        used in the computations.

    smooth_gain : float
        Smoothness constrain gain

    debug : bool, optional
        If True, print debugging information.

    gradient : bool, optional
        If True, the gradient of the morphing function is returned.

    Returns
    -------

    penalty or  gradient values.

    penalty : float
        Value of the cost function

    gradient_values : ndarray (float64 ,ndim = 3), optional
        If gradient keyword is True, the gradient of the function is also
        returned.

    """

    sector_displacement_2d = \
        sector_displacement_1d.reshape(*((2,) + tuple(blocks_shape)))

    if input_images.shape[0] == 3:
        three_times = True
        previous_image = input_images[0]
        center_image = input_images[1]
        next_image = input_images[2]

    else:
        previous_image = None
        center_image = input_images[0]
        next_image = input_images[1]
        three_times = False

    if gradient:
        gradient_values = _cost_function(sector_displacement_2d,
                                         center_image,
                                         next_image,
                                         mask,
                                         smooth_gain,
                                         gradient=True)
        if three_times:
            gradient_values += _cost_function(sector_displacement_2d,
                                              previous_image,
                                              center_image,
                                              mask,
                                              smooth_gain,
                                              gradient=True)

        return gradient_values.ravel()

    else:
        residuals, smoothness_penalty = _cost_function(sector_displacement_2d,
                                                       center_image,
                                                       next_image,
                                                       mask,
                                                       smooth_gain,
                                                       gradient=False)

        if three_times:
            _residuals, _smoothness = _cost_function(sector_displacement_2d,
                                                     previous_image,
                                                     center_image,
                                                     mask,
                                                     smooth_gain,
                                                     gradient=False)

            residuals += _residuals
            smoothness_penalty += _smoothness

        if debug:
            print("\nresiduals", residuals)
            print("smoothness_penalty", smoothness_penalty)

        return residuals + smoothness_penalty


def vet(input_images,
        sectors=((32, 16, 4, 2), (32, 16, 4, 2)),
        smooth_gain=1e6,
        first_guess=None,
        intermediate_steps=False,
        verbose=True,
        indexing='yx',
        padding=0,
        options=None):
    """
    Variational Echo Tracking Algorithm presented in
    `Laroche and Zawadzki (1995)`_  and used in the McGill Algorithm for
    Prediction by Lagrangian Extrapolation (MAPLE) described in
    `Germann and Zawadzki (2002)`_.

    .. _`Laroche and Zawadzki (1995)`:\
        http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    .. _`Germann and Zawadzki (2002)`:\
        http://dx.doi.org/10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2

    This algorithm computes the displacement field between two images
    ( the input_image with respect to the template image).
    The displacement is sought by minimizing sum of the residuals of the
    squared differences of the images pixels and the contribution of a
    smoothness constrain.

    In order to find the minimum an scaling guess procedure is applied,
    from larger scales
    to a finer scale. This reduces the changes that the minimization procedure
    converges to a local minimum. The scaling guess is defined by the scaling
    sectors (see **sectors** keyword).

    The smoothness of the returned displacement field is controlled by the
    smoothness constrain gain (**smooth_gain** keyword).

    If a first guess is not given, zero displacements are used as first guess.

    To minimize the cost function, the `scipy minimization`_ function is used
    with the 'CG' method. This method proved to give the best results under
    any different conditions and is the most similar one to the original VET
    implementation in `Laroche and Zawadzki (1995)`_.


    The method CG uses a nonlinear conjugate gradient algorithm by Polak and
    Ribiere, a variant of the Fletcher-Reeves method described in
    Nocedal and Wright (2006), pp. 120-122.

    .. _`scipy minimization`: \
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------

    input_images : ndarray_ or MaskedArray
        Input images, sequence of 2D arrays, or 3D arrays.
        The first dimension represents the images time dimension.

        The template_image (first element in first dimensions) denotes the
        reference image used to obtain the displacement (2D array).
        The second is the target image.

        The expected dimensions are (2,ni,nj).

    sectors : list or array, optional
        The number of sectors for each dimension used in the scaling procedure.
        If dimension is 1, the same sectors will be used both image dimensions
        (x and y). If is 2D, the each row determines the sectors of the
        each dimension.

    smooth_gain : float, optional
        Smooth gain factor

    first_guess : ndarray_, optional
        The shape of the first guess should have the same shape as the initial
        sectors shapes used in the scaling procedure.
        If first_guess is not present zeros are used as first guess.

        E.g.:
            If the first sector shape in the scaling procedure is (ni,nj), then
            the first_guess should have (2, ni, nj ) shape.

    intermediate_steps : bool, optional
        If True, also return a list with the first guesses obtained during the
        scaling procedure. False, by default.

    verbose : bool, optional
        Verbosity enabled if True (default).

    indexing : str, optional
        Input indexing order.'ij' and 'xy' indicates that the
        dimensions of the input are (time, longitude, latitude), while
        'yx' indicates (time, latitude, longitude).
        The displacement field dimensions are ordered accordingly in a way that
        the first dimension indicates the displacement along x (0) or y (1).
        That is, UV displacements are always returned.

    padding : int
        Padding width in grid points. A border is added to the input array
        to reduce the effects of the minimization at the border.

    options : dict, optional
        A dictionary of solver options.
        See `scipy minimization`_ function for more details.

    Returns
    -------

    displacement_field : ndarray_
        Displacement Field (2D array representing the transformation) that
        warps the template image into the input image.
        The dimensions are (2,ni,nj), where the first
        dimension indicates the displacement along x (0) or y (1).

    intermediate_steps : list of ndarray_
        List with the first guesses obtained during the scaling procedure.

    References
    ----------

    Laroche, S., and I. Zawadzki, 1995:
    Retrievals of horizontal winds from single-Doppler clear-air data by
    methods of cross-correlation and variational analysis.
    J. Atmos. Oceanic Technol., 12, 721–738.
    doi: http://dx.doi.org/10.1175/1520-0426(1995)012<0721:ROHWFS>2.0.CO;2

    Germann, U. and I. Zawadzki, 2002:
    Scale-Dependence of the Predictability of Precipitation from Continental
    Radar Images.  Part I: Description of the Methodology.
    Mon. Wea. Rev., 130, 2859–2873,
    doi: 10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2.

    Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.

    """

    if verbose:
        def debug_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def debug_print(*args, **kwargs):
            del args
            del kwargs

    if options is None:
        options = dict()
    else:
        options = dict(options)

    options.setdefault('eps', 0.1)
    options.setdefault('gtol', 0.1)
    options.setdefault('maxiter', 100)
    options.setdefault('disp', False)

    # Set to None to suppress pylint warning.
    pad_i = None
    pad_j = None
    sectors_in_i = None
    sectors_in_j = None

    debug_print("Running VET algorithm")

    if (input_images.ndim != 3) or (1 < input_images.shape[0] > 3):
        raise ValueError("input_images dimension mismatch.\n" +
                         "input_images.shape: " + str(input_images.shape) +
                         "\n(2, x, y ) or (2, x, y ) dimensions expected")

    valid_indexing = ['yx', 'xy', 'ij']

    if indexing not in valid_indexing:
        raise ValueError("Invalid indexing valus: {0}\n".format(indexing)
                         + "Supported values: {0}".format(str(valid_indexing)))

    # Get mask
    if isinstance(input_images, MaskedArray):
        mask = numpy.ma.getmaskarray(input_images)
    else:
        # Mask invalid data
        if padding > 0:
            padding_tuple = ((0, 0), (padding, padding), (padding, padding))

            input_images = numpy.pad(input_images,
                                     padding_tuple,
                                     'constant',
                                     constant_values=numpy.nan)

        input_images = numpy.ma.masked_invalid(input_images)
        mask = numpy.ma.getmaskarray(input_images)

    input_images[mask] = 0  # Remove any Nan from the raw data

    # Create a 2D mask with the right data type for _vet
    mask = numpy.asarray(numpy.any(mask, axis=0), dtype='int8')

    input_images = numpy.asarray(input_images.data, dtype='float64')

    # Check that the sectors divide the domain
    sectors = numpy.asarray(sectors, dtype="int")

    if sectors.ndim == 1:

        new_sectors = (numpy.zeros((2,) + sectors.shape, dtype='int')
                       + sectors.reshape((1, sectors.shape[0]))
                       )
        sectors = new_sectors
    elif sectors.ndim > 2 or sectors.ndim < 1:
        raise ValueError("Incorrect sectors dimensions.\n"
                         + "Only 1D or 2D arrays are supported to define"
                         + "the number of sectors used in"
                         + "the scaling procedure")

    # Sort sectors in descending order
    sectors[0, :].sort()
    sectors[1, :].sort()

    # Prepare first guest
    first_guess_shape = (2, int(sectors[0, 0]), int(sectors[1, 0]))

    if first_guess is None:
        first_guess = numpy.zeros(first_guess_shape, order='C')
    else:
        if first_guess.shape != first_guess_shape:
            raise ValueError(
                "The shape of the initial guess do not match the number of "
                + "sectors of the first scaling guess\n"
                + "first_guess.shape={}\n".format(str(first_guess.shape))
                + "Expected shape={}".format(str(first_guess_shape)))
        else:
            first_guess = numpy.asarray(first_guess, order='C', dtype='float64')

    scaling_guesses = list()

    previous_sectors_in_i = sectors[0, 0]
    previous_sectors_in_j = sectors[1, 0]

    for n, (sectors_in_i, sectors_in_j) in enumerate(zip(sectors[0, :],
                                                         sectors[1, :])):

        # Minimize for each sector size
        pad_i = get_padding(input_images.shape[1], sectors_in_i)
        pad_j = get_padding(input_images.shape[2], sectors_in_j)

        if (pad_i != (0, 0)) or (pad_j != (0, 0)):

            _input_images = numpy.pad(input_images, ((0, 0), pad_i, pad_j), 'edge')

            _mask = numpy.pad(mask, (pad_i, pad_j),
                              'constant',
                              constant_values=1)

            if first_guess is None:
                first_guess = numpy.pad(first_guess, ((0, 0), pad_i, pad_j), 'edge')
        else:
            _input_images = input_images
            _mask = mask

        sector_shape = (_input_images.shape[1] // sectors_in_i,
                        _input_images.shape[2] // sectors_in_j)

        debug_print("original image shape: " + str(_input_images.shape))
        debug_print("padded image shape: " + str(_input_images.shape))
        debug_print("padded template_image image shape: "
                    + str(_input_images.shape))

        debug_print("\nNumber of sectors: {0:d},{1:d}".format(sectors_in_i,
                                                              sectors_in_j))

        debug_print("Sector Shape:", sector_shape)

        if n > 0:
            first_guess = zoom(first_guess,
                               (1,
                                sectors_in_i / previous_sectors_in_i,
                                sectors_in_j / previous_sectors_in_j),
                               order=1, mode='nearest')

        debug_print("Minimizing")

        result = minimize(vet_cost_function,
                          first_guess.flatten(),
                          jac=vet_cost_function_gradient,
                          args=(_input_images,
                                (sectors_in_i, sectors_in_j),
                                _mask,
                                smooth_gain),
                          method='CG',
                          options=options)

        first_guess = result.x.reshape(*first_guess.shape)

        if verbose:
            vet_cost_function(result.x,
                              _input_images,
                              (sectors_in_i, sectors_in_j),
                              _mask,
                              smooth_gain,
                              debug=True)
        if indexing == 'yx':
            scaling_guesses.append(first_guess[::-1, ...])
        else:
            scaling_guesses.append(first_guess)

        previous_sectors_in_i = sectors_in_i
        previous_sectors_in_j = sectors_in_j

    first_guess = zoom(first_guess,
                       (1,
                        _input_images.shape[1] / sectors_in_i,
                        _input_images.shape[2] / sectors_in_j),
                       order=1, mode='nearest')

    # Remove the extra padding if any
    ni = _input_images.shape[1]
    nj = _input_images.shape[2]

    first_guess = first_guess[:, pad_i[0]:ni - pad_i[1], pad_j[0]:nj - pad_j[1]]

    if indexing == 'yx':
        first_guess = first_guess[::-1, ...]

    if padding > 0:
        first_guess = first_guess[:, padding:-padding, padding:-padding]

    if intermediate_steps:
        return first_guess, scaling_guesses

    return first_guess
