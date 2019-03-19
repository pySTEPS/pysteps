"""
pysteps.timeseries.autoregression
=================================

Methods related to autoregressive AR(p) models.

.. autosummary::
    :toctree: ../generated/

    adjust_lag2_corrcoef1
    adjust_lag2_corrcoef2
    ar_acf
    estimate_ar_params_yw
    iterate_ar_model
"""

import numpy as np

def adjust_lag2_corrcoef1(gamma_1, gamma_2):
    """A simple adjustment of lag-2 temporal autocorrelation coefficient to
    ensure that the resulting AR(2) process is stationary when the parameters
    are estimated from the Yule-Walker equations.

    Parameters
    ----------
    gamma_1 : float
      Lag-1 temporal autocorrelation coeffient.
    gamma_2 : float
      Lag-2 temporal autocorrelation coeffient.

    Returns
    -------
    out : float
      The adjusted lag-2 correlation coefficient.

    """
    gamma_2 = max(gamma_2, 2*gamma_1*gamma_1-1+1e-10)
    gamma_2 = min(gamma_2, 1-1e-10)

    return gamma_2


def adjust_lag2_corrcoef2(gamma_1, gamma_2):
    """A more advanced adjustment of lag-2 temporal autocorrelation coefficient
    to ensure that the resulting AR(2) process is stationary when the parameters
    are estimated from the Yule-Walker equations.

    Parameters
    ----------
    gamma_1 : float
      Lag-1 temporal autocorrelation coeffient.
    gamma_2 : float
      Lag-2 temporal autocorrelation coeffient.

    Returns
    -------
    out : float
      The adjusted lag-2 correlation coefficient.

    """
    gamma_2 = max(gamma_2, 2*gamma_1*gamma_2-1)
    gamma_2 = max(gamma_2, (3*gamma_1**2-2+2*(1-gamma_1**2)**1.5) / gamma_1**2)

    return gamma_2


def ar_acf(gamma, n=None):
    """Compute theoretical autocorrelation function (ACF) from the AR(p) model
    with lag-l, l=1,2,...,p temporal autocorrelation coefficients.

    Parameters
    ----------
    gamma : array-like
      Array of length p containing the lag-l, l=1,2,...p, temporal autocorrelation
      coefficients. The correlation coefficients are assumed to be in ascending
      order with respect to time lag.
    n : int
      Desired length of ACF array. Must be greater than len(gamma).

    Returns
    -------
    out : array-like
      Array containing the ACF values.

    """
    ar_order = len(gamma)
    if n == ar_order or n is None:
        return gamma
    elif n < ar_order:
        raise ValueError("n=%i, but must be larger than the order of the AR process %i" % (n,ar_order))

    phi = estimate_ar_params_yw(gamma)[:-1]

    acf = gamma.copy()
    for t in range(0, n - ar_order):
        # Retrieve gammas (in reverse order)
        gammas = acf[t:t + ar_order][::-1]
        # Compute next gamma
        gamma_ = np.sum(gammas*phi)
        acf.append(gamma_)

    return acf


def estimate_ar_params_yw(gamma):
    """Estimate the parameters of an AR(p) model from the Yule-Walker equations
    using the given set of autocorrelation coefficients.

    Parameters
    ----------
    gamma : array_like
      Array of length p containing the lag-l, l=1,2,...p, temporal autocorrelation
      coefficients. The correlation coefficients are assumed to be in ascending
      order with respect to time lag.

    Returns
    -------
    out : ndarray
      An array of shape (n,p+1) containing the AR(p) parameters for for the
      lag-p terms for each cascade level, and also the standard deviation of
      the innovation term.

    """
    p = len(gamma)

    phi = np.empty(p+1)

    g = np.hstack([[1.0], gamma])
    G = []
    for j in range(p):
        G.append(np.roll(g[:-1], j))
    G = np.array(G)
    phi_ = np.linalg.solve(G, g[1:].flatten())

    # Check that the absolute values of the roots of the characteristic
    # polynomial are less than one. Otherwise the AR(p) model is not stationary.
    r = np.array([np.abs(r_) for r_ in np.roots([1.0 if i == 0 else -phi_[i] \
                  for i in range(p)])])
    if any(r >= 1):
        raise RuntimeError(
            "Error in estimate_ar_params_y: "
            "nonstationary AR(p) process")

    c = 1.0
    for j in range(p):
      c -= gamma[j] * phi_[j]
    phi_pert = np.sqrt(c)

    # If the expression inside the square root is negative, phi_pert cannot
    # be computed and it is set to zero instead.
    if not np.isfinite(phi_pert):
        phi_pert = 0.0

    phi[:p] = phi_
    phi[-1] = phi_pert

    return phi


def iterate_ar_model(X, phi, EPS=None):
    """Apply an AR(p) model to a time-series of two-dimensional fields.

    Parameters
    ----------
    X : array_like
      Three-dimensional array of shape (p,w,h) containing a time series of p
      two-dimensional fields of shape (w,h). The fields are assumed to be in
      ascending order by time, and the timesteps are assumed to be regular.
    phi : array_like
      Array of length p+1 specifying the parameters of the AR(p) model. The
      parameters are in ascending order by increasing time lag, and the last
      element is the parameter corresponding to the innovation term EPS.
    EPS : array_like
      Optional perturbation field for the AR(p) process. If EPS is None, the
      innovation term is not added.

    """
    if X.shape[0] != len(phi)-1:
      raise ValueError("dimension mismatch between X and phi: X.shape[0]=%d, len(phi)=%d" % (X.shape[0], len(phi)))

    if EPS is not None and EPS.shape != (X.shape[1], X.shape[2]):
        raise ValueError("dimension mismatch between X and EPS: X.shape=%s, EPS.shape=%s" % (str(X.shape), str(EPS.shape)))

    X_new = 0.0

    p = len(phi) - 1

    for i in range(p):
        X_new += phi[i] * X[-(i+1), :, :]

    if EPS is not None:
        X_new += phi[-1] * EPS

    return np.stack(list(X[1:, :, :]) + [X_new])
