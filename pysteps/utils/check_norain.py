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
