# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.linear_blending
================================

Implementation of the linear blending between nowcast and NWP data.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import numpy as np
from pysteps import nowcasts
from pysteps.utils import transformation


def forecast(
    precip,
    velocity,
    timesteps,
    timestep,
    nowcast_method,
    R_nwp=None,
    start_blending=120,
    end_blending=240,
    nowcast_kwargs=dict(),
):

    """Generate a forecast by linearly blending nowcasts with NWP data

    Parameters
    ----------
    precip: array_like
      Array containing the input precipitation field(s) ordered by timestamp
      from oldest to newest. The time steps between the inputs are assumed
      to be regular.
    velocity; array_like
      Array of shape (2, m, n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    timesteps: int
      Number of time steps to forecast.
    timestep: int or float
      The time difference (in minutes) between consecutive forecast fields.
    nowcast_method: {'anvil', 'eulerian', 'lagrangian', 'extrapolation', 'lagrangian_probability', 'sprog', 'steps', 'sseps'}
      Name of the nowcast method used to forecast the precipitation.
    R_nwp: array_like or NoneType, optional
      Array of shape (timesteps, m, n) in the case of no ensemble or
      of shape (n_ens_members, timesteps, m, n) in the case of an ensemble
      containing the NWP precipitation fields ordered by timestamp from oldest
      to newest. The time steps between the inputs are assumed to be regular
      (and identical to the time step between the nowcasts). If no NWP
      data is given the value of R_nwp is None and no blending will be performed.
    start_blending: int, optional
      Time stamp (in minutes) after which the blending should start. Before this
      only the nowcast data is used.
    end_blending: int, optional
      Time stamp (in minutes) after which the blending should end. Between
      start_blending and end_blending the nowcasts and NWP data are linearly
      merged with each other. After end_blending only the NWP data is used.
    nowcast_kwargs: dict, optional
      Dictionary containing keyword arguments for the nowcast method.

    Returns
    -------
    R_blended: ndarray
      Array of shape (timesteps, m, n) in the case of no ensemble or
      of shape (n_ens_members, timesteps, m, n) in the case of an ensemble
      containing the precipation forecast generated by linearly blending
      the nowcasts and the NWP data.
    """

    # Calculate the nowcasts
    nowcast_method_func = nowcasts.get_method(nowcast_method)
    R_nowcast = nowcast_method_func(
        precip,
        velocity,
        timesteps,
        **nowcast_kwargs,
    )

    # Transform the precipitation back to mm/h
    R_nowcast = transformation.dB_transform(R_nowcast, threshold=-10.0, inverse=True)[0]

    # Check if NWP data is given as input
    if R_nwp is not None:

        if len(R_nowcast.shape) == 4:
            n_ens_members_nowcast = R_nowcast.shape[0]
            if n_ens_members_nowcast == 1:
                R_nowcast = np.squeeze(R_nowcast)
        else:
            n_ens_members_nowcast = 1

        if len(R_nwp.shape) == 4:
            n_ens_members_nwp = R_nwp.shape[0]
            if n_ens_members_nwp == 1:
                R_nwp = np.squeeze(R_nwp)
        else:
            n_ens_members_nwp = 1

        n_ens_members_max = max(n_ens_members_nowcast, n_ens_members_nwp)
        n_ens_members_min = min(n_ens_members_nowcast, n_ens_members_nwp)

        if n_ens_members_min != n_ens_members_max:
            if n_ens_members_nwp == 1:
                R_nwp = np.repeat(R_nwp[np.newaxis, :, :], n_ens_members_max, axis=0)
            elif n_ens_members_nowcast == 1:
                R_nowcast = np.repeat(
                    R_nowcast[np.newaxis, :, :], n_ens_members_max, axis=0
                )
            else:
                repeats = [
                    (n_ens_members_max + i) // n_ens_members_min
                    for i in range(n_ens_members_min)
                ]

                if n_ens_members_nwp == n_ens_members_min:
                    R_nwp = np.repeat(R_nwp, repeats, axis=0)
                elif n_ens_members_nowcast == n_ens_members_min:
                    R_nowcast = np.repeat(R_nowcast, repeats, axis=0)

        # Check if dimensions are correct
        assert (
            R_nwp.shape == R_nowcast.shape
        ), "The dimensions of R_nowcast and R_nwp need to be identical: dimension of R_nwp = {} and dimension of R_nowcast = {}".format(
            R_nwp.shape, R_nowcast.shape
        )

        # Initialise output
        R_blended = np.zeros_like(R_nowcast)

        # Calculate the weights
        for i in range(timesteps):
            # Calculate what time we are at
            t = (i + 1) * timestep

            # Calculate the weight with a linear relation (weight_nwp at start_blending = 0.0)
            # and (weight_nwp at end_blending = 1.0)
            weight_nwp = (t - start_blending) / (end_blending - start_blending)

            # Set weights at times before start_blending and after end_blending
            if weight_nwp < 0.0:
                weight_nwp = 0.0
            elif weight_nwp > 1.0:
                weight_nwp = 1.0

            # Calculate weight_nowcast
            weight_nowcast = 1.0 - weight_nwp

            # Calculate output by combining R_nwp and R_nowcast,
            # while distinguishing between ensemble and non-ensemble methods
            if n_ens_members_max == 1:
                R_blended[i, :, :] = (
                    weight_nwp * R_nwp[i, :, :] + weight_nowcast * R_nowcast[i, :, :]
                )
            else:
                R_blended[:, i, :, :] = (
                    weight_nwp * R_nwp[:, i, :, :]
                    + weight_nowcast * R_nowcast[:, i, :, :]
                )

            # Find where the NaN values are and replace them with NWP data
            nan_indices = np.isnan(R_blended)
            R_blended[nan_indices] = R_nwp[nan_indices]
    else:
        # If no NWP data is given, the blended field is simply equal to the nowcast field
        R_blended = R_nowcast

    return R_blended
