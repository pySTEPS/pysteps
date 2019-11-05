"""
pysteps.nowcasts.utils
======================

Module with common utilities used by nowcasts methods.

.. autosummary::
    :toctree: ../generated/

    print_ar_params
    print_corrcoefs
    stack_cascades
    recompose_cascade
"""

import numpy as np


def print_ar_params(PHI):
    """Print the parameters of an AR(p) model.

    Parameters
    ----------
    PHI : array_like
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
    GAMMA : array_like
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


def stack_cascades(R_d, n_levels, donorm=True, convert_to_full_arrays=False):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d : list
      List of cascades obtained by calling a method implemented in
      pysteps.cascade.decomposition.
    n_levels : int
      The number of cascade levels.
    donorm : bool
      If True, normalize the cascade levels before stacking.

    Returns
    -------
    out : tuple
      A three-element tuple containing 1) a list of three-dimensional arrays
      containing the rearranged cascade levels and 2) lists of mean values and
      3) standard deviations for each cascade level (taken from the last element
      in R_d).
    """
    R_c = []
    mu = np.empty(n_levels)
    sigma = np.empty(n_levels)

    n_inputs = len(R_d)

    for i in range(n_levels):
        R_ = []
        mu_ = 0
        sigma_ = 1
        for j in range(n_inputs):
            if donorm:
                mu_ = R_d[j]["means"][i]
                sigma_ = R_d[j]["stds"][i]
            R__ = (R_d[j]["cascade_levels"][i] - mu_) / sigma_
            if R_d[j]["compact_output"] and convert_to_full_arrays:
                R_tmp = np.zeros(R_d[j]["weight_masks"].shape[1:], dtype=complex)
                R_tmp[R_d[j]["weight_masks"][i]] = R__
                R__ = R_tmp
            R_.append(R__)
        mu[i] = R_d[n_inputs - 1]["means"][i]
        sigma[i] = R_d[n_inputs - 1]["stds"][i]
        R_c.append(np.stack(R_))

    if not np.any([R_d[i]["compact_output"] for i in range(len(R_d))]):
        R_c = np.stack(R_c)

    return R_c, mu, sigma
