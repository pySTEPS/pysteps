"""
pysteps.utils.arrays
====================

Utility methods for creating and processing arrays.

.. autosummary::
    :toctree: ../generated/

    compute_centred_coord_array
"""

import numpy as np


def compute_centred_coord_array(height, width):
    """Compute a 2D coordinate array, where the origin is at the center.

    Parameters
    ----------
    height : int
      The height of the array.
    width : int
      The width of the array.

    Returns
    -------
    out : tuple of ndarray
      The two coordinate arrays.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """

    if height % 2 == 1:
        s1 = np.s_[-int(height / 2) : int(height / 2) + 1]
    else:
        s1 = np.s_[-int(height / 2) : int(height / 2)]

    if width % 2 == 1:
        s2 = np.s_[-int(width / 2) : int(width / 2) + 1]
    else:
        s2 = np.s_[-int(width / 2) : int(width / 2)]

    y_coordinates, x_coordinates = np.ogrid[s1, s2]

    return y_coordinates, x_coordinates
