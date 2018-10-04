"""Implementation of the semi-Lagrangian method of Germann et al (2002)."""

import numpy as np
import scipy.ndimage.interpolation as ip
import time

def extrapolate(R, V, num_timesteps, outval=np.nan, **kwargs):
    """Apply semi-Lagrangian extrapolation to a two-dimensional precipitation
    field.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) containing the input precipitation field. All
        values are required to be finite.
    V : array-like
        Array of shape (2,m,n) containing the x- and y-components of the m*n
        advection field. All values are required to be finite.
    num_timesteps : int
        Number of time steps to extrapolate.
    outval : float
        Optional argument for specifying the value for pixels advected from
        outside the domain. If outval is set to 'min', the value is taken as
        the minimum value of R.
        Default : np.nan

    Other Parameters
    ----------------
    D_prev : array-like
        Optional initial displacement vector field of shape (2,m,n) for the
        extrapolation.
        Default : None
    n_iter : int
        Number of inner iterations in the semi-Lagrangian scheme.
        Default : 3
    inverse : bool
        If True, the extrapolation trajectory is computed backward along the
        flow (default), forward otherwise.
        Default : True
    return_displacement : bool
        If True, return the total advection velocity (displacement) between the
        initial input field and the advected one integrated along the trajectory.
        Default : False

    Returns
    -------
    out : array or tuple
        If return_displacement=False, return a time series extrapolated fields of
        shape (num_timesteps,m,n). Otherwise, return a tuple containing the
        extrapolated fields and the total displacement along the advection trajectory.

    References
    ----------
    :cite:`GZ2002`

    """
    if len(R.shape) != 2:
        raise ValueError("R must be a two-dimensional array")

    if len(V.shape) != 3:
        raise ValueError("V must be a three-dimensional array")

    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    # defaults
    verbose             = kwargs.get("verbose", False)
    D_prev              = kwargs.get("D_prev", None)
    n_iter              = kwargs.get("n_iter", 3)
    inverse             = kwargs.get("inverse", True)
    return_displacement = kwargs.get("return_displacement", False)

    if verbose:
        print("Computing the advection with the semi-lagrangian scheme.")
        t0 = time.time()

    if outval == "min":
        outval = np.nanmin(R)

    coeff = 1.0 if not inverse else -1.0

    X,Y = np.meshgrid(np.arange(V.shape[2]), np.arange(V.shape[1]))
    XY  = np.stack([X, Y])

    R_e = []
    if D_prev is None:
        D = np.zeros((2, V.shape[1], V.shape[2]))
    else:
        D = D_prev.copy()

    for t in range(num_timesteps):
        V_inc = np.zeros(D.shape)

        for k in range(n_iter):
            if t > 0 or k > 0 or D_prev is not None:
                XYW = XY + D - V_inc / 2.0
                XYW = [XYW[1, :, :], XYW[0, :, :]]

                VWX = ip.map_coordinates(V[0, :, :], XYW, mode="nearest", order=0,
                                        prefilter=False)
                VWY = ip.map_coordinates(V[1, :, :], XYW, mode="nearest", order=0,
                                        prefilter=False)
            else:
                VWX = V[0, :, :]
                VWY = V[1, :, :]

            V_inc[0, :, :] = VWX / n_iter
            V_inc[1, :, :] = VWY / n_iter

            D += coeff * V_inc

        XYW = XY + D
        XYW = [XYW[1, :, :], XYW[0, :, :]]

        IW = ip.map_coordinates(R, XYW, mode="constant", cval=outval, order=0,
                              prefilter=False)
        R_e.append(np.reshape(IW, R.shape))

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    if not return_displacement:
        return np.stack(R_e)
    else:
        return np.stack(R_e), D
