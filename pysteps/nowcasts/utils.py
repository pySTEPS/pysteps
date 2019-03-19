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


def stack_cascades(R_d, n_levels, donorm=True):
    """Stack the given cascades into a larger array.
    
    Parameters
    ----------
    R_d : list
      List of cascades obtained by calling a method implemented in
      pysteps.cascade.decomposition.
    n_levels : int
      Number of cascade levels.
    donorm : bool
      If True, normalize the cascade levels before stacking.
    
    Returns
    -------
    out : tuple
      A three-element tuple containing a four-dimensional array of stacked
      cascade levels and lists of mean values and standard deviations for each
      cascade level (taken from the last cascade).
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
            if j == n_inputs - 1:
                mu[i] = mu_
                sigma[i] = sigma_
            R__ = (R_d[j]["cascade_levels"][i, :, :] - mu_) / sigma_
            R_.append(R__)
        R_c.append(np.stack(R_))

    return np.stack(R_c), mu, sigma


def recompose_cascade(R, mu, sigma):
    """Recompose a cascade by inverting the normalization and summing the
    cascade levels.
    
    Parameters
    ----------
    R : array_like
      
    """
    R_rc = [(R[i, :, :] * sigma[i]) + mu[i] for i in range(len(mu))]
    R_rc = np.sum(np.stack(R_rc), axis=0)

    return R_rc