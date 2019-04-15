"""
pysteps.motion.constant
=======================

Implementation of a constant advection field estimation by maximizing the
correlation between two images.

.. autosummary::
    :toctree: ../generated/

    constant

"""

import numpy as np
import scipy.ndimage.interpolation as ip
import scipy.optimize as op

def constant(R, **kwargs):
    """Compute a constant advection field by finding a translation vector that
    maximizes the correlation between two successive images.

    Parameters
    ----------
    R : array_like
      Array of shape (T,m,n) containing a sequence of T two-dimensional input
      images of shape (m,n). If T > 2, two last elements along axis 0 are used.
    """
    X, Y = np.meshgrid(np.arange(R.shape[2]), np.arange(R.shape[1]))

    def f(v):
        XYW = [Y + v[1], X + v[0]]
        R_w = ip.map_coordinates(R[-2, :, :], XYW, mode="constant", cval=np.nan,
                                 order=0, prefilter=False)

        mask = np.isfinite(R_w)

        return -np.corrcoef(R[-1, :, :][mask], R_w[mask])[0, 1]

    options = {"initial_simplex" : (np.array([(0, 1), (1, 0), (1, 1)]))}
    result = op.minimize(f, (1, 1), method="Nelder-Mead", options=options)

    return -result.x
