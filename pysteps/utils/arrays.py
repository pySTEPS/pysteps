"""
pysteps.utils.arrays
====================

Utility methods for creating and processing arrays.

.. autosummary::
    :toctree: ../generated/

    compute_centred_coord_array
"""

import numpy as np


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
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC
