"""
pysteps.utils.arrays
====================

Utility methods for creating and processing arrays.

.. autosummary::
    :toctree: ../generated/

    compute_centred_coord_array
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def block_reduce(image, block_size, func=np.mean, **kwargs):
    """Downsample image by applying a function to local blocks
    (average by default).
    If `image` is not perfectly divisible by `block_size` along a
    given axis, the data will be trimmed (from the end) along that
    axis to make it perfectly divisible.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an `axis` parameter.
        By default, `numpy.mean` is used.
    block_size : array_like or int
        Array containing down-sampling integer factor along each axis.
        If an integer value is given, the same block shape is used for all the
        image dimensions.

    Other Parameters
    ----------------

    kwargs : dict
        Parameters passed to the `func` used to calculate the return value for
        each local block.


    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
    """
    # Code adapted from Scikit-image. This function merges the
    # skimage.measure.block_reduce and skimage.util.view_as_blocks functions
    # to avoid an extra dependency in pysteps.
    #
    # Copyright (C) 2019, the scikit-image team

    if isinstance(block_size, int):
        block_size = (block_size,) * image.ndim

    block_shape = np.array(block_size)

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    if (block_shape <= 0).any():
        raise ValueError("'block_size' elements must be strictly positive")

    sel_slice = list()
    for i in range(len(block_size)):
        values_to_clip = image.shape[i] % block_size[i]
        if values_to_clip == 0:
            sel_slice.append(slice(None))
        else:
            sel_slice.append(slice(0, -values_to_clip))

    image = image[sel_slice]

    image_shape = np.array(image.shape)

    # Re-stride the array to build the block view

    new_shape = tuple(image_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(image.strides * block_shape) + image.strides

    new_image = as_strided(image, shape=new_shape, strides=new_strides)

    return func(new_image, axis=tuple(range(image.ndim, new_image.ndim)),
                **kwargs)


def compute_centred_coord_array(M, N):
    """Compute a 2D coordinate array, where the origin is at the center.

    Parameters
    ----------
    M : int
      The height of the array.
    N : int
      The width of the array.

    Returns
    -------
    out : ndarray
      The coordinate array.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2):int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2):int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2):int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2):int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC
