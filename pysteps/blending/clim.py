"""
pysteps.blending.clim
=====================

Module with methods to read, write and compute past and climatological model weights.

.. autosummary::
    :toctree: ../generated/

    get_default_weights    
    save_weights
    calc_clim_weights
"""

import numpy as np
from os.path import exists, join
import pickle
from pysteps import rcparams


def get_default_weights(n_cascade_levels=8, n_models=1):
    """
    Get the default weights as given in BPS2004.
    Take subset of n_cascade_levels or add entries with small values (1e-4) if
    n_cascade_levels differs from 8.

    Parameters
    ----------
    n_cascade_levels: int, optional
      Number of cascade levels.
    n_models: int, optional
      Number of NWP models

    Returns
    -------
    default_weights: array-like
      Array of shape [model, scale_level] containing the climatological weights.

    """

    default_weights = np.array(
        [0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040]
    )
    n_weights = default_weights.shape[0]
    if n_cascade_levels < n_weights:
        default_weights = default_weights[0:n_cascade_levels]
    elif n_cascade_levels > n_weights:
        default_weights = np.append(
            default_weights, np.repeat(1e-4, n_cascade_levels - n_weights)
        )
    return np.resize(default_weights, (n_models, n_cascade_levels))


def save_weights(
    current_weights,
    validtime,
    outdir_path=rcparams.outputs["path_workdir"],
    window_length=30,
):
    """
    Add the current NWP weights to update today's daily average weight. If the
    day is over, update the list of daily average weights covering a rolling
    window.

    Parameters
    ----------
    current_weights: array-like
      Array of shape [model, scale_level, ...]
      containing the current weights of the different NWP models per cascade level.
    outdir_path: string
      Path to folder where the historical weights are stored.
    window_length: int, optional
      Length of window (in days) over which to compute the climatological weights.

    Returns
    -------
    None

    """

    n_cascade_levels = current_weights.shape[1]

    # Load weights_today, a dictionary containing {mean_weights, n, last_validtime}
    weights_today_file = join(outdir_path, "NWP_weights_today.pkl")
    if exists(weights_today_file):
        with open(weights_today_file, "rb") as f:
            weights_today = pickle.load(f)
    else:
        weights_today = {
            "mean_weights": np.copy(current_weights),
            "n": 0,
            "last_validtime": validtime,
        }

    # Load the past weights which is an array with dimensions day x model x scale_level
    past_weights_file = join(outdir_path, "NWP_weights_window.npy")
    past_weights = None
    if exists(past_weights_file):
        past_weights = np.load(past_weights_file)
    # First check if we have started a new day wrt the last written weights, in which
    # case we should update the daily weights file and reset daily statistics.
    if weights_today["last_validtime"].date() < validtime.date():
        # Append weights to the list of the past X daily averages.
        if past_weights is not None:
            past_weights = np.append(
                past_weights, [weights_today["mean_weights"]], axis=0
            )
        else:
            past_weights = np.array([weights_today["mean_weights"]])
        print(past_weights.shape)
        # Remove oldest if the number of entries exceeds the window length.
        if past_weights.shape[0] > window_length:
            past_weights = np.delete(past_weights, 0, axis=0)
        # FIXME also write out last_validtime.date() in this file?
        # In that case it will need pickling or netcdf.
        # Write out the past weights within the rolling window.
        np.save(past_weights_file, past_weights)
        # Reset statistics for today.
        weights_today["n"] = 0
        weights_today["mean_weights"] = 0

    # Reset today's weights if needed and/or compute online average from the
    # current weights using numerically stable algorithm
    weights_today["n"] += 1
    weights_today["mean_weights"] += (
        current_weights - weights_today["mean_weights"]
    ) / weights_today["n"]
    weights_today["last_validtime"] = validtime
    with open(weights_today_file, "wb") as f:
        pickle.dump(weights_today, f)

    return None


def calc_clim_weights(
    n_cascade_levels=8,
    outdir_path=rcparams.outputs["path_workdir"],
    nmodels=1,
    window_length=30,
):
    """
    Return the climatological weights based on the daily average weights in the
    rolling window. This is done using a geometric mean.

    Parameters
    ----------
    n_cascade_levels: int, optional
      Number of cascade levels.
    outdir_path: string, optional
      Path to folder where the historical weights are stored.
    nmodels: int, optional
      Number of NWP models
    window_length: int, optional
      Length of window (in days) over which to compute the climatological weights.

    Returns
    -------
    climatological_mean_weights: array-like
      Array of shape [model, scale_level, ...] containing the climatological weights.

    """
    past_weights_file = join(outdir_path, "NWP_weights_window.npy")
    # past_weights has dimensions date x model x scale_level  x ....
    if exists(past_weights_file):
        past_weights = np.load(past_weights_file)
    else:
        past_weights = None
    # check if there's enough data to compute the climatological skill
    if not past_weights or past_weights.shape[0] < window_length:
        return get_default_weights(n_cascade_levels, nmodels)
    # reduce window if necessary
    else:
        past_weights = past_weights[-window_length:]

    # Calculate climatological weights from the past_weights using the
    # geometric mean.
    geomean_weights = np.exp(np.log(past_weights).mean(axis=0))

    return geomean_weights
