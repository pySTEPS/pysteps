"""
pysteps.blending.clim
=====================

Module with methods to read, write and compute past and climatological model skill.

.. autosummary::
    :toctree: ../generated/

    save_weights
    calc_clim_weights
"""

import numpy as np


def save_weights(current_weights, validtime, model_names, outdir_path, window_length):
    """
    Add the current NWP weights/skills to update today's daily average weight. If
    the day is over, update the list of daily average weights covering a
    rolling window.

    Parameters
    ----------
    current_weights: array-like
      Array of shape [model, scale_level, ...]
      containing the current weights of the different NWP models per cascade level.
    model_names : list of strings
      List containing unique identifiers for each NWP model.
    outdir_path: string
      Path to folder where the historical weights are stored.
    window_length: int
      Length of window over which to compute the climatological weights (in days).

    Returns
    -------
    None

    """

    # TODO everywhere:  error catching

    # Load weights_today, a dictionary containing {mean_weights, n, last_validtime}
    weights_today = np.load(outdir_path + "NWP_weights_today.bin")

    # Load the past weights which is an array with dimensions day x model x scale_level
    past_weights = np.load(outdir_path + "NWP_weights_window.bin")

    # First check if we have started a new day wrt the last written weights, in which
    # case we should update the daily weights file and reset daily statistics.
    if weights_today.last_validtime.date() < validtime.date():
        # Append weights to the list of the past X daily averages.
        past_weights = np.append(past_weights, [weights_today.mean_weights], axis=0)
        # Remove oldest if the number of entries exceeds the window length.
        if past_weights.shape[0] > window_length:
            past_weights = np.delete(past_weights, 0, axis=0)
        # TODO also write out last_validtime.date() in this file?
        # In that case it will need pickling or netcdf.
        # Write out the past weights within the rolling window.
        np.save(outdir_path + "NWP_weights_window.bin", past_weights)
        # Reset statistics for today.
        weights_today.n = 0
        weights_today.mean_weights = 0

    # Reset today's weights if needed and/or compute online average from the
    # current weights using numerically stable algorithm
    weights_today.n += 1
    weights_today.mean_weights += (
        current_weights - weights_today.mean_weights
    ) / weights_today.n
    weights_today.last_validtime = validtime

    np.save(outdir_path + "NWP_weights_today.bin", weights_today, allow_pickle=True)

    return None


def calc_clim_weights(model_names, outdir_path):
    """
    Return the climatological skill based on the weights in the rolling
    window. This is done using a geometric mean.

    Parameters
    ----------
    weights: array-like
      Array of shape [model, scale_level, ...]
      containing the current weights of the different NWP models per cascade level.
    model_names : list of strings
      List containing unique identifiers for each NWP model.
    outdir_path: string
      Path to folder where the historical weights are stored.
    window_length: int
      Length of window over which to compute the climatological weights (in days).

    Returns
    -------
    climatological_mean_weights: array-like
      Array containing the climatological weights.

    """

    past_weights = np.load(outdir_path + "NWP_weights_window.bin")

    # Calculate climatological weights from the past_weights using the
    # geometric mean.
    geomean_weights = np.exp(np.log(past_weights).mean(axis=0))

    return geomean_weights
