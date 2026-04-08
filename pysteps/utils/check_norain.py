import numpy as np

from pysteps import utils


def check_norain(precip_arr, precip_thr=None, norain_thr=0.0, win_fun=None):
    """

    Parameters
    ----------
    precip_arr:  array-like
      An at least 2 dimensional array containing the input precipitation field
    precip_thr: float, optional
      Specifies the threshold value for minimum observable precipitation intensity. If None, the
      minimum value over the domain is taken.
    norain_thr: float, optional
      Specifies the threshold value for the fraction of rainy pixels in precip_arr below which we consider there to be
      no rain. Standard set to 0.0
    win_fun: {'hann', 'tukey', None}
      Optional tapering function to be applied to the input field, generated with
      :py:func:`pysteps.utils.tapering.compute_window_function`
      (default None).
      This parameter needs to match the window function you use in later noise generation,
      or else this method will say that there is rain, while after the tapering function is
      applied there is no rain left, so you will run into a ValueError.
    Returns
    -------
    norain: bool
      Returns whether the fraction of rainy pixels is below the norain_thr threshold.

    """

    if win_fun is not None:
        tapering = utils.tapering.compute_window_function(
            precip_arr.shape[-2], precip_arr.shape[-1], win_fun
        )
    else:
        tapering = np.ones((precip_arr.shape[-2], precip_arr.shape[-1]))

    tapering_mask = tapering == 0.0
    masked_precip = precip_arr.copy()
    masked_precip[..., tapering_mask] = np.nanmin(precip_arr)

    if precip_thr is None:
        precip_thr = np.nanmin(masked_precip)
    rain_pixels = masked_precip[masked_precip > precip_thr]
    norain = rain_pixels.size / masked_precip.size <= norain_thr
    print(
        f"Rain fraction is: {str(rain_pixels.size / masked_precip.size)}, while minimum fraction is {str(norain_thr)}"
    )
    return norain


def check_previous_radar_obs(precip, ar_order):
    """Check all radar time steps and remove zero precipitation and constant field time steps

    Parameters
    ----------
    precip : array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the
      inputs are assumed to be regular.
    ar_order : int
      The order of the autoregressive model to use. Must be >= 1.

    Returns
    -------
    precip : numpy array
      Array of shape (ar_order+1,m,n) containing the modified array with
      input precipitation fields ordered by timestamp from oldest to newest.
      The time steps between the inputs are assumed to be regular.
    ar_order : int
      The order of the autoregressive model to use. Must be >= 1.
      Adapted to match with precip.shape equal (ar_order+1,m,n).
    """
    if not precip.shape[0] >= 2:
        raise ValueError(
            "Wrong precip shape. The radar input must have at least 2 time steps."
        )

    # Check all time steps for zero-precip/constant field (constant field, minimum==maximum)
    const_precip = [np.nanmin(obs) == np.nanmax(obs) for obs in precip]
    # Check the cases
    if const_precip[-1] or ~np.any(const_precip):
        # Unchanged if no rain in latest time step -> will be processed as zero_precip_radar=True
        # or Unchanged if all time steps contain rain
        return precip, ar_order
    elif const_precip[-2]:
        # This case means radar-observed rain in the latest but no rain in the 2nd latest time steps.
        # Solution 1:
        # Assume the precipitation means clutter / parasit echoes in this case.
        # Treat it as a default zero-precip case, AR-2 model
        precip = np.ones((3, precip.shape[1], precip.shape[2])) * np.nanmin(precip)
        # Give a warning
        print(
            "\n[WARNING] Precip + no-precip cases in the 2 latest radar input time steps.\nCannot calculate autoregression. Set to zero-precip radar input.\n"
        )
        return precip, 2
        # # Solution 2 (to be discussed):
        # # Adjust the radar input precipitation (if possible)
        # # try to use a previous time step
        # if not np.all(zero_precip[:-2]):
        #     # find latest non-zero precip
        #     # ATTENTION: This changes the time between precip[-2] and precip[-1] from initial 5min to a longer period
        #     print(
        #         "[WARNING] Radar input time steps adapted and ar_order set to 1. Input delta time changed."
        #     )
        #     prev = np.arange(len(zero_precip[:-2]))[~np.array(zero_precip[:-2])][-1]
        #     # Adjust the time between input time steps to match the modified precip array
        #     ### read the deltatime between the radar time step from metadata in current_deltatime
        #     ### new_deltatime = (precip.shape[0] - prev) * current_deltatime
        #     ### code here to set new_deltatimes the radar time step delta time in metadata
        #     return precip[[prev, -1]], 1
        # raise ValueError(
        #     "Precipitation in latest but no previous time step. Not possible to calculate autoregression."
        # )
    else:
        # Keep the latest time steps that do all contain precip
        precip = precip[np.max(np.arange(len(const_precip))[const_precip]) + 1 :].copy()
        if precip.shape[0] - 1 < ar_order:
            # Give a warning
            print(
                f"[WARNING] Radar input only with {precip.shape[0]} non-zero time steps and ar_order set to {precip.shape[0]-1}."
            )
        return precip, min(ar_order, precip.shape[0] - 1)
