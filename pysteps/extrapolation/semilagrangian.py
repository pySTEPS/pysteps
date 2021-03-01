# -*- coding: utf-8 -*-
"""
pysteps.extrapolation.semilagrangian
====================================

Implementation of the semi-Lagrangian method described in :cite:`GZ2002`.

.. autosummary::
    :toctree: ../generated/

    extrapolate

"""

import time
import warnings

import numpy as np
import scipy.ndimage.interpolation as ip


def extrapolate(
    precip,
    velocity,
    timesteps,
    outval=np.nan,
    xy_coords=None,
    allow_nonfinite_values=False,
    vel_timestep=1,
    **kwargs,
):
    """Apply semi-Lagrangian backward extrapolation to a two-dimensional
    precipitation field.

    Parameters
    ----------
    precip: array-like or None
        Array of shape (m,n) containing the input precipitation field. All
        values are required to be finite by default. If set to None, only the
        displacement field is returned without interpolating the inputs. This
        requires that return_displacement is set to True.
    velocity: array-like
        Array of shape (2,m,n) containing the x- and y-components of the m*n
        advection field. All values are required to be finite by default.
    timesteps: int or list of floats
        If timesteps is integer, it specifies the number of time steps to
        extrapolate. If a list is given, each element is the desired
        extrapolation time step from the current time. The elements of the list
        are required to be in ascending order.
    outval: float, optional
        Optional argument for specifying the value for pixels advected from
        outside the domain. If outval is set to 'min', the value is taken as
        the minimum value of precip.
        Default: np.nan
    xy_coords: ndarray, optional
        Array with the coordinates of the grid dimension (2, m, n ).

        * xy_coords[0]: x coordinates
        * xy_coords[1]: y coordinates

        By default, the *xy_coords* are computed for each extrapolation.
    allow_nonfinite_values: bool, optional
        If True, allow non-finite values in the precipitation and advection
        fields. This option is useful if the input fields contain a radar mask
        (i.e. pixels with no observations are set to nan).

    Other Parameters
    ----------------
    displacement_prev: array-like
        Optional initial displacement vector field of shape (2,m,n) for the
        extrapolation.
        Default: None
    n_iter: int
        Number of inner iterations in the semi-Lagrangian scheme. If n_iter > 0,
        the integration is done using the midpoint rule. Otherwise, the advection
        vectors are taken from the starting point of each interval.
        Default: 1
    return_displacement: bool
        If True, return the displacement between the initial input field and
        the one obtained by integrating along the advection field.
        Default: False
    vel_timestep: float
        The time step of the velocity field. It is assumed to have the same
        unit as the timesteps argument. Applicable if timeseps is a list.
        Default: 1.
    interp_order: int
        The order of interpolation to use. Default: 1 (linear). Setting this
        to 0 (nearest neighbor) gives the best computational performance but
        may produce visible artefacts. Setting this to 3 (cubic) gives the best
        ability to reproduce small-scale variability but may significantly
        increase the computation time.

    Returns
    -------
    out: array or tuple
        If return_displacement=False, return a time series extrapolated fields
        of shape (num_timesteps,m,n). Otherwise, return a tuple containing the
        extrapolated fields and the integrated trajectory (displacement) along
        the advection field.


    References
    ----------
    :cite:`GZ2002`
    """

    if precip is not None and precip.ndim != 2:
        raise ValueError("precip must be a two-dimensional array")

    if velocity.ndim != 3:
        raise ValueError("velocity must be a three-dimensional array")

    if not allow_nonfinite_values:
        if precip is not None and np.any(~np.isfinite(precip)):
            raise ValueError("precip contains non-finite values")

        if np.any(~np.isfinite(velocity)):
            raise ValueError("velocity contains non-finite values")

    if precip is not None and np.all(~np.isfinite(precip)):
        raise ValueError("precip contains only non-finite values")

    if np.all(~np.isfinite(velocity)):
        raise ValueError("velocity contains only non-finite values")

    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")

    # defaults
    verbose = kwargs.get("verbose", False)
    displacement_prev = kwargs.get("displacement_prev", None)
    n_iter = kwargs.get("n_iter", 1)
    return_displacement = kwargs.get("return_displacement", False)
    interp_order = kwargs.get("interp_order", 1)

    if precip is None and not return_displacement:
        raise ValueError("precip is None but return_displacement is False")

    if "D_prev" in kwargs.keys():
        warnings.warn(
            "deprecated argument D_prev is ignored, use displacement_prev instead",
        )

    # if interp_order > 1, apply separate masking to preserve nan and
    # non-precipitation values
    if precip is not None and interp_order > 1:
        minval = np.nanmin(precip)
        mask_min = (precip > minval).astype(float)
        if allow_nonfinite_values:
            mask_finite = np.isfinite(precip)
            precip = precip.copy()
            precip[~mask_finite] = 0.0
            mask_finite = mask_finite.astype(float)

    prefilter = True if interp_order > 1 else False

    if isinstance(timesteps, int):
        timesteps = np.arange(1, timesteps + 1)
        vel_timestep = 1.0
    elif np.any(np.diff(timesteps) <= 0.0):
        raise ValueError("the given timestep sequence is not monotonously increasing")

    timestep_diff = np.hstack([[timesteps[0]], np.diff(timesteps)])

    if verbose:
        print("Computing the advection with the semi-lagrangian scheme.")
        t0 = time.time()

    if precip is not None and outval == "min":
        outval = np.nanmin(precip)

    if xy_coords is None:
        x_values, y_values = np.meshgrid(
            np.arange(velocity.shape[2]), np.arange(velocity.shape[1])
        )

        xy_coords = np.stack([x_values, y_values])

    def interpolate_motion(displacement, velocity_inc, td):
        coords_warped = xy_coords + displacement
        coords_warped = [coords_warped[1, :, :], coords_warped[0, :, :]]

        velocity_inc_x = ip.map_coordinates(
            velocity[0, :, :], coords_warped, mode="nearest", order=1, prefilter=False
        )
        velocity_inc_y = ip.map_coordinates(
            velocity[1, :, :], coords_warped, mode="nearest", order=1, prefilter=False
        )

        velocity_inc[0, :, :] = velocity_inc_x
        velocity_inc[1, :, :] = velocity_inc_y

        if n_iter > 1:
            velocity_inc /= n_iter

        velocity_inc *= td / vel_timestep

    precip_extrap = []
    if displacement_prev is None:
        displacement = np.zeros((2, velocity.shape[1], velocity.shape[2]))
        velocity_inc = velocity.copy() * timestep_diff[0] / vel_timestep
    else:
        displacement = displacement_prev.copy()
        velocity_inc = np.empty(velocity.shape)
        interpolate_motion(displacement, velocity_inc, timestep_diff[0])

    for ti, td in enumerate(timestep_diff):
        if n_iter > 0:
            for k in range(n_iter):
                interpolate_motion(displacement - velocity_inc / 2.0, velocity_inc, td)
                displacement -= velocity_inc
                interpolate_motion(displacement, velocity_inc, td)
        else:
            if ti > 0 or displacement_prev is not None:
                interpolate_motion(displacement, velocity_inc, td)

            displacement -= velocity_inc

        coords_warped = xy_coords + displacement
        coords_warped = [coords_warped[1, :, :], coords_warped[0, :, :]]

        if precip is not None:
            precip_warped = ip.map_coordinates(
                precip,
                coords_warped,
                mode="constant",
                cval=outval,
                order=interp_order,
                prefilter=prefilter,
            )

            if interp_order > 1:
                mask_warped = ip.map_coordinates(
                    mask_min,
                    coords_warped,
                    mode="constant",
                    cval=0,
                    order=1,
                    prefilter=False,
                )
                precip_warped[mask_warped < 0.5] = minval

                if allow_nonfinite_values:
                    mask_warped = ip.map_coordinates(
                        mask_finite,
                        coords_warped,
                        mode="constant",
                        cval=0,
                        order=1,
                        prefilter=False,
                    )
                    precip_warped[mask_warped < 0.5] = np.nan

            precip_extrap.append(np.reshape(precip_warped, precip.shape))

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    if precip is not None:
        if not return_displacement:
            return np.stack(precip_extrap)
        else:
            return np.stack(precip_extrap), displacement
    else:
        return None, displacement
