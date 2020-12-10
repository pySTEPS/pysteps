# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.utils
======================

Module with common utilities used by nowcasts methods.

.. autosummary::
    :toctree: ../generated/

    print_ar_params
    print_corrcoefs
    stack_cascades
"""

import numpy as np


def binned_timesteps(timesteps):
    """Compute a binning of the given irregular time steps.

    Parameters
    ----------
    timesteps: array_like
        List or one-dimensional array containing the time steps in ascending
        order.

    Returns
    -------
    out: list
        List of length int(np.ceil(timesteps[-1]))+1 containing the bins. Each
        element is a list containing the indices of the time steps falling in
        the bin (excluding the right edge).
    """
    timesteps = list(timesteps)
    if not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")

    if np.any(np.array(timesteps) < 0):
        raise ValueError("negative time steps are not allowed")

    num_bins = int(np.ceil(timesteps[-1]))
    timestep_range = np.arange(num_bins + 1)
    bin_idx = np.digitize(timesteps, timestep_range, right=False)

    out = [[] for i in range(num_bins + 1)]
    for i, bi in enumerate(bin_idx):
        out[bi - 1].append(i)

    return out


def print_ar_params(PHI):
    """Print the parameters of an AR(p) model.

    Parameters
    ----------
    PHI: array_like
        Array of shape (n, p) containing the AR(p) parameters for n cascade
        levels.
    """
    print("****************************************")
    print("* AR(p) parameters for cascade levels: *")
    print("****************************************")

    n = PHI.shape[1]

    hline_str = "---------"
    for k in range(n):
        hline_str += "---------------"

    print(hline_str)
    title_str = "| Level |"
    for k in range(n - 1):
        title_str += "    Phi-%d     |" % (k + 1)
    title_str += "    Phi-0     |"
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for k in range(n):
        fmt_str += " %-12.6f |"

    for k in range(PHI.shape[0]):
        print(fmt_str % ((k + 1,) + tuple(PHI[k, :])))
        print(hline_str)


def print_corrcoefs(GAMMA):
    """Print the parameters of an AR(p) model.

    Parameters
    ----------
    GAMMA: array_like
      Array of shape (m, n) containing n correlation coefficients for m cascade
      levels.
    """
    print("************************************************")
    print("* Correlation coefficients for cascade levels: *")
    print("************************************************")

    m = GAMMA.shape[0]
    n = GAMMA.shape[1]

    hline_str = "---------"
    for k in range(n):
        hline_str += "----------------"

    print(hline_str)
    title_str = "| Level |"
    for k in range(n):
        title_str += "     Lag-%d     |" % (k + 1)
    print(title_str)
    print(hline_str)

    fmt_str = "| %-5d |"
    for k in range(n):
        fmt_str += " %-13.6f |"

    for k in range(m):
        print(fmt_str % ((k + 1,) + tuple(GAMMA[k, :])))
        print(hline_str)


def stack_cascades(R_d, n_levels, convert_to_full_arrays=False):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d: list
        List of cascades obtained by calling a method implemented in
        pysteps.cascade.decomposition.
    n_levels: int
        The number of cascade levels.

    Returns
    -------
    out: tuple
        A list of three-dimensional arrays containing the stacked cascade levels.
    """
    R_c = []

    n_inputs = len(R_d)

    for i in range(n_levels):
        R_ = []
        for j in range(n_inputs):
            R__ = R_d[j]["cascade_levels"][i]
            if R_d[j]["compact_output"] and convert_to_full_arrays:
                R_tmp = np.zeros(R_d[j]["weight_masks"].shape[1:], dtype=complex)
                R_tmp[R_d[j]["weight_masks"][i]] = R__
                R__ = R_tmp
            R_.append(R__)
        R_c.append(np.stack(R_))

    if not np.any([R_d[i]["compact_output"] for i in range(len(R_d))]):
        R_c = np.stack(R_c)

    return R_c
