"""
pysteps.blending.clim
=====================

Module with methods to read, write and compute past and climatological NWP model
skill scores. The module stores the average daily skill score for the past t days
and updates it every day. The resulting average climatological skill score is
the skill the NWP model skill regresses to during the blended forecast. If no
climatological values are present, the default skill from :cite:`BPS2006` is used.

.. autosummary::
    :toctree: ../generated/

    get_default_skill
    save_skill
    calc_clim_skill
"""

import pickle
from pathlib import Path

import numpy as np


def get_default_skill(n_cascade_levels=8, n_models=1):
    """
    Get the default climatological skill values as given in :cite:`BPS2006`.
    Take subset of n_cascade_levels or add entries with small values (1e-4) if
    n_cascade_levels differs from 8.

    Parameters
    ----------
    n_cascade_levels: int, optional
      Number of cascade levels. Defaults to 8.
    n_models: int, optional
      Number of NWP models. Defaults to 1.

    Returns
    -------
    default_skill: array-like
      Array of shape [model, scale_level] containing the climatological skill
      values.

    """

    default_skill = np.array(
        [0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040]
    )
    n_skill = default_skill.shape[0]
    if n_cascade_levels < n_skill:
        default_skill = default_skill[0:n_cascade_levels]
    elif n_cascade_levels > n_skill:
        default_skill = np.append(
            default_skill, np.repeat(1e-4, n_cascade_levels - n_skill)
        )
    return np.resize(default_skill, (n_models, n_cascade_levels))


def save_skill(
    current_skill,
    validtime,
    outdir_path,
    window_length=30,
    **kwargs,
):
    """
    Add the current NWP skill to update today's daily average skill. If the day
    is over, update the list of daily average skill covering a rolling window.

    Parameters
    ----------
    current_skill: array-like
      Array of shape [model, scale_level, ...]
      containing the current skill of the different NWP models per cascade
      level.
    validtime: datetime
      Datetime object containing the date and time for which the current
      skill are valid.
    outdir_path: string
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams.
    window_length: int, optional
      Length of window (in days) of daily skill that should be retained.
      Defaults to 30.

    Returns
    -------
    None

    """

    n_cascade_levels = current_skill.shape[1]

    # Load skill_today, a dictionary containing {mean_skill, n, last_validtime}
    new_skill_today_file = False

    skill_today_file = Path(outdir_path) / "NWP_skill_today.pkl"
    if skill_today_file.is_file():
        with open(skill_today_file, "rb") as f:
            skill_today = pickle.load(f)
        if skill_today["mean_skill"].shape != current_skill.shape:
            new_skill_today_file = True
    else:
        new_skill_today_file = True

    if new_skill_today_file:
        skill_today = {
            "mean_skill": np.copy(current_skill),
            "n": 0,
            "last_validtime": validtime,
        }

    # Load the past skill which is an array with dimensions day x model x scale_level
    past_skill_file = Path(outdir_path) / "NWP_skill_window.npy"
    past_skill = None
    if past_skill_file.is_file():
        past_skill = np.load(past_skill_file)
    # First check if we have started a new day wrt the last written skill, in which
    # case we should update the daily skill file and reset daily statistics.
    if skill_today["last_validtime"].date() < validtime.date():
        # Append skill to the list of the past X daily averages.
        if (
            past_skill is not None
            and past_skill.shape[2] == n_cascade_levels
            and past_skill.shape[1] == skill_today["mean_skill"].shape[0]
        ):
            past_skill = np.append(past_skill, [skill_today["mean_skill"]], axis=0)
        else:
            past_skill = np.array([skill_today["mean_skill"]])

        # Remove oldest if the number of entries exceeds the window length.
        if past_skill.shape[0] > window_length:
            past_skill = np.delete(past_skill, 0, axis=0)
        # FIXME also write out last_validtime.date() in this file?
        # In that case it will need pickling or netcdf.
        # Write out the past skill within the rolling window.
        np.save(past_skill_file, past_skill)
        # Reset statistics for today.
        skill_today["n"] = 0
        skill_today["mean_skill"] = 0

    # Reset today's skill if needed and/or compute online average from the
    # current skill using numerically stable algorithm
    skill_today["n"] += 1
    skill_today["mean_skill"] += (
        current_skill - skill_today["mean_skill"]
    ) / skill_today["n"]
    skill_today["last_validtime"] = validtime
    # Make path if path does not exist
    skill_today_file.parent.mkdir(exist_ok=True, parents=True)
    # Open and write to skill file
    with open(skill_today_file, "wb") as f:
        pickle.dump(skill_today, f)

    return None


def calc_clim_skill(
    outdir_path,
    n_cascade_levels=8,
    n_models=1,
    window_length=30,
):
    """
    Return the climatological skill based on the daily average skill in the
    rolling window. This is done using a geometric mean.

    Parameters
    ----------
    n_cascade_levels: int, optional
      Number of cascade levels.
    outdir_path: string
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams.
    n_models: int, optional
      Number of NWP models. Defaults to 1.
    window_length: int, optional
      Length of window (in days) over which to compute the climatological
      skill. Defaults to 30.

    Returns
    -------
    climatological_mean_skill: array-like
      Array of shape [model, scale_level, ...] containing the climatological
      (geometric) mean skill.

    """
    past_skill_file = Path(outdir_path) / "NWP_skill_window.npy"
    # past_skill has dimensions date x model x scale_level  x ....
    if past_skill_file.is_file():
        past_skill = np.load(past_skill_file)
    else:
        past_skill = np.array(None)
    # check if there is enough data to compute the climatological skill
    if not past_skill.any():
        return get_default_skill(n_cascade_levels, n_models)
    elif past_skill.shape[0] < window_length:
        return get_default_skill(n_cascade_levels, n_models)
    # reduce window if necessary
    else:
        past_skill = past_skill[-window_length:]

    # Make sure past_skill cannot be lower than 10e-5
    past_skill = np.where(past_skill < 10e-5, 10e-5, past_skill)

    # Calculate climatological skill from the past_skill using the
    # geometric mean.
    geomean_skill = np.exp(np.log(past_skill).mean(axis=0))

    # Make sure skill is always a positive value and a finite value
    geomean_skill = np.where(geomean_skill < 10e-5, 10e-5, geomean_skill)
    geomean_skill = np.nan_to_num(
        geomean_skill, copy=True, nan=10e-5, posinf=10e-5, neginf=10e-5
    )

    return geomean_skill
