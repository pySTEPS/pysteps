"""
pysteps.timeseries.autoregression
=================================

Methods related to autoregressive AR(p) models.

.. autosummary::
    :toctree: ../generated/

    adjust_lag2_corrcoef1
    adjust_lag2_corrcoef2
    ar_acf
    estimate_ar_params_ols
    estimate_ar_params_yw
    estimate_var_params_ols
    estimate_var_params_yw
    iterate_ar_model
    iterate_var_model
"""

import numpy as np
from scipy.special import binom
from scipy import linalg as la


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
    to ensure that the resulting AR(2) process is stationary when
    the parameters are estimated from the Yule-Walker equations.

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
        Array of length p containing the lag-l, l=1,2,...p, temporal
        autocorrelation coefficients.
        The correlation coefficients are assumed to be in ascending
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


def estimate_ar_params_ols(x, p, d=0, check_stationarity=True,
                           include_constant_term=False, h=0, lam=0.0):
    """Estimate the parameters of an autoregressive integrated AR(p) model

    :math:`x_{k+1}=c+\phi_1 x_k+\phi_2 x_{k-1}+\dots+\phi_p x_{k-p}+\phi_{p+1}\epsilon`

    by using ordinary least squares (OLS). If :math:`d\geq 1`, the parameters
    are estimated for a d times differenced time series that is integrated back
    to the original one by summation of the differences.

    Parameters
    ----------
    x : array_like
        Array of shape (n,...) containing a time series of length n>=p+d+h+1.
        The remaining dimensions are flattened.
    p : int
        The order of the model.
    d : {0,1}
        The order of differencing to apply to the time series.
    check_stationarity : bool
        Check the stationarity of the estimated model.
    include_constant_term : bool
        Include the constant term c to the model.
    h : int
        If h>0, the fitting is done by using a history of length h in addition
        to the minimal required number of time steps n=p+d+1.
    lam : float
        If lam>0, the regression is regularized by adding a penalty term
        (i.e. ridge regression).

    Returns
    -------
    out : list
        The estimated parameter matrices :math:`\mathbf{\Phi}_1,\mathbf{\Phi}_2,
        \dots,\mathbf{\Phi}_{p+1}`. If include_constant_term is True, the
        constant term :math:`\mathbf{c}` is added to the beginning of the list.
    """
    n = x.shape[0]

    if n < p + d + h + 1:
        raise ValueError("n = %d, p = %d, d = %d, but n >= p+d+h+1 required" % (n, p, d))

    if len(x.shape) > 1:
        x = x.reshape((n, np.prod(x.shape[1:])))

    if d >= 1:
        x = np.diff(x, axis=0)
        n -= d

    x_lhs = x[p:, :]

    Z = []
    for i in range(x.shape[1]):
        for j in range(p-1, n-1-h):
            z_ = np.hstack([x[j-k, i] for k in range(p)])
            if include_constant_term:
                z_ = np.hstack([[1], z_])
            Z.append(z_)
    Z = np.column_stack(Z)

    b = np.dot(np.dot(x_lhs, Z.T), np.linalg.inv(np.dot(Z, Z.T) + lam*np.eye(Z.shape[0])))
    b = b.flatten()

    phi = []
    if include_constant_term:
        c = b[0]
        phi = b[1:]
    else:
        phi = b
    phi = np.hstack([phi, [0.0]])

    if check_stationarity:
        if not test_ar_stationarity(phi[:-1]):
            raise RuntimeError(
                "Error in estimate_ar_params_yw: "
                "nonstationary AR(p) process")

    if d == 1:
        phi_out = []
        for i in range(p+d+1):
            phi_out.append(0.0)

        for i in range(1, d+1):
            phi_out[i-1] -= binom(d, i) * (-1)**i
        for i in range(1, p+1):
            phi_out[i-1] += phi[i-1]
        for i in range(1, p+1):
            for j in range(1, d+1):
                phi_out[i+j-1] += phi[i-1] * binom(d, j) * (-1)**j

        if include_constant_term:
            return c, phi_out
        else:
            return phi_out
    else:
        if include_constant_term:
            return c, phi
        else:
            return phi


def estimate_ar_params_yw(gamma, check_stationarity=True):
    """Estimate the parameters of an AR(p) model

    :math:`x_{k+1}=\phi_1 x_k+\phi_2 x_{k-1}+\dots+\phi_p x_{k-p}+\phi_{p+1}\epsilon`

    from the Yule-Walker equations using the given set of autocorrelation
    coefficients.

    Parameters
    ----------
    gamma : array_like
        Array of length p containing the lag-l temporal autocorrelation
        coefficients for l=1,2,...p. The correlation coefficients are assumed
        to be in ascending order with respect to time lag.
    check_stationarity : bool
        If True, the stationarity of the resulting VAR(p) process is tested. An
        exception is thrown if the process is not stationary.

    Returns
    -------
    out : ndarray
        Array of length p+1 containing the AR(p) parameters for for the
        lag-p terms and the innovation term.

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
    # polynomial are less than one.
    # Otherwise the AR(p) model is not stationary.
    if check_stationarity:
        if not test_ar_stationarity(phi_):
            raise RuntimeError(
                "Error in estimate_ar_params_yw: "
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


def estimate_var_params_ols(x, p, d=0, check_stationarity=True,
                            include_constant_term=False, h=0, lam=0.0):
    """Estimate the parameters of a vector autoregressive integrated VAR(p) model

      :math:`\mathbf{x}_{k+1}=\mathbf{c}+\mathbf{\Phi}_1\mathbf{x}_k+
      \mathbf{\Phi}_2\mathbf{x}_{k-1}+\dots+\mathbf{\Phi}_p\mathbf{x}_{k-p}+
      \mathbf{\Phi}_{p+1}\mathbf{\epsilon}`

    by using ordinary least squares (OLS). If :math:`d\geq 1`, the parameters
    are estimated for a d times differenced time series that is integrated back
    to the original one by summation of the differences.

    Parameters
    ----------
    x : array_like
        Array of shape (q,n,:) containing a time series of length n with
        q-dimensional variables. The remaining dimensions are flattened.
    p : int
        The order of the model.
    d : {0,1}
        The order of differencing to apply to the time series.
    check_stationarity : bool
        Check the stationarity of the estimated model.
    include_constant_term : bool
        Include the constant term c to the model.
    h : int
        If h>0, the fitting is done by using a history of length h in addition
        to the minimal required number of time steps n=p+d+1.
    lam : float
        If lam>0, the regression is regularized by adding a penalty term
        (i.e. ridge regression).

    Returns
    -------
    out : list
        The estimated parameter matrices :math:`\mathbf{\Phi}_1,\mathbf{\Phi}_2,
        \dots,\mathbf{\Phi}_{p+1}`. If include_constant_term is True, the
        constant term :math:`\mathbf{c}` is added to the beginning of the list.
    """
    q = x.shape[0]
    n = x.shape[1]

    if n < p + d + h + 1:
        raise ValueError("n = %d, p = %d, d = %d, but n >= p+d+h+1 required" % (n, p, d))

    if d >= 1:
        x = np.diff(x, axis=1)
        n -= d

    x = x.reshape((q, n, np.prod(x.shape[2:])))

    X = []
    for i in range(x.shape[2]):
        for j in range(p+h, n):
            x_ = x[:, j, i]
            X.append(x_.reshape((q, 1)))
    X = np.hstack(X)

    Z = []
    for i in range(x.shape[2]):
        for j in range(p-1, n-1-h):
            z_ = np.vstack([x[:, j-k, i].reshape((q, 1)) for k in range(p)])
            if include_constant_term:
                z_ = np.vstack([[1], z_])
            Z.append(z_)
    Z = np.hstack(Z)

    B = np.dot(np.dot(X, Z.T), np.linalg.inv(np.dot(Z, Z.T) + lam*np.eye(Z.shape[0])))

    phi = []
    if include_constant_term:
        c = B[:, 0]
        for i in range(p):
            phi.append(B[:, i*q+1:(i+1)*q+1])
    else:
        for i in range(p):
            phi.append(B[:, i*q:(i+1)*q])
    phi.append(np.zeros((q, q)))

    if check_stationarity:
        M = np.zeros((p*q, p*q))

        for i in range(p):
            M[0:q, i*q:(i+1)*q] = phi[i]
        for i in range(1, p):
            M[i*q:(i+1)*q, (i-1)*q:i*q] = np.eye(q, q)
        r, v = np.linalg.eig(M)

        if np.any(np.abs(r) > 0.999):
            raise RuntimeError(
                "Error in estimate_var_params_ols: "
                "nonstationary VAR(p) process")

    if d == 1:
        phi_out = []
        for i in range(p+d):
            phi_out.append(np.zeros((q, q)))
        phi_out.append(np.zeros((q, q)))

        for i in range(1, d+1):
            phi_out[i-1] -= binom(d, i) * (-1)**i * np.eye(q, q)
        for i in range(1, p+1):
            phi_out[i-1] += phi[i-1]
        for i in range(1, p+1):
            for j in range(1, d+1):
                phi_out[i+j-1] += phi[i-1] * binom(d, j) * (-1)**j

        if include_constant_term:
            return c, phi_out
        else:
            return phi_out
    else:
        if include_constant_term:
            return c, phi
        else:
            return phi

def estimate_var_params_yw(gamma, d=0, check_stationarity=True):
    """Estimate the parameters of a VAR(p) model

      :math:`\mathbf{x}_{k+1}=\mathbf{\Phi}_1\mathbf{x}_k+
      \mathbf{\Phi}_2\mathbf{x}_{k-1}+\dots+\mathbf{\Phi}_p\mathbf{x}_{k-p}+
      \mathbf{\Phi}_{p+1}\mathbf{\epsilon}`

    from the Yule-Walker equations using the given correlation matrices
    :math:`\mathbf{\Gamma}_0,\mathbf{\Gamma}_1,\dots,\mathbf{\Gamma}_n`, where
    n=p.

    Parameters
    ----------
    gamma : list
        List of correlation matrices
        :math:`\mathbf{\Gamma}_0,\mathbf{\Gamma}_1,\dots,\mathbf{\Gamma}_n`.
        See :py:func:`pysteps.timeseries.correlation.temporal_autocorrelation_multivariate`.
    d : int
        The order of differencing. If d>=1, a differencing operator
        :math:`\Delta=(1-L)^d`, where :math:`L` is a time lag operator, is
        applied to produce parameter estimates for a vector autoregressive
        integrated VARI(p,d) model of order d.
    check_stationarity : bool
        If True, the stationarity of the resulting VAR(p) process is tested. An
        exception is thrown if the process is not stationary.

    Returns
    -------
    out : list
        List of VAR(p) coefficient matrices :math:`\mathbf{\Phi}_1,
        \mathbf{\Phi}_2,\dots\mathbf{\Phi}_{p+1}`, where the last matrix
        corresponds to the innovation term.

    Notes
    -----
    To estimate the parameters of a VARI(p,d) model, call
    correlation.temporal_autocorrelation_multivariate with d>0.
    """
    p = len(gamma) - 1
    q = gamma[0].shape[0]

    for i in range(len(gamma)):
        if gamma[i].shape[0] != q or gamma[i].shape[1] != q:
            raise ValueError("dimension mismatch: gamma[%d].shape=%s, but (%d,%d) expected" % \
                             (i, str(gamma[i].shape, q, q)))

    a = np.empty((p*q, p*q))
    for i in range(p):
        for j in range(p):
            a_tmp = gamma[abs(i-j)]
            if i > j:
                a_tmp = a_tmp.T
            a[i*q:(i+1)*q, j*q:(j+1)*q] = a_tmp

    b = np.vstack([gamma[i].T for i in range(1, p+1)])
    x = np.linalg.solve(a, b)

    phi = []
    for i in range(p):
        phi.append(x[i*q:(i+1)*q, :])

    if check_stationarity:
        if not test_var_stationarity(phi):
            raise RuntimeError(
                "Error in estimate_var_params_yw: "
                "nonstationary VAR(p) process")

    if d >= 1:
        phi_out = []
        for i in range(p+d):
            phi_out.append(np.zeros((q, q)))

        for i in range(1, d+1):
            phi_out[i-1] -= binom(d, i) * (-1)**i * np.eye(q, q)
        for i in range(1, p+1):
            phi_out[i-1] += phi[i-1]
        for i in range(1, p+1):
            for j in range(1, d+1):
                phi_out[i+j-1] += phi[i-1] * binom(d, j) * (-1)**j
    else:
        phi_out = phi

    phi_stacked = np.hstack(phi)
    phi_out.append(la.sqrtm(gamma[0] - np.dot(np.dot(phi_stacked, a), phi_stacked.T)))

    if d >= 1:
        return phi_out
    else:
        return phi


def iterate_ar_model(x, phi, eps=None):
    """Apply an AR(p) model

    :math:`x_{k+1}=\phi_1 x_k+\phi_2 x_{k-1}+\dots+\phi_p x_{k-p}+\phi_{p+1}\epsilon`

    to a time series :math:`x_k`.

    Parameters
    ----------
    x : array_like
        Array of shape (n,...), n>=p, containing a time series of a input variable
        x. The elements of x along the first dimension are assumed to be in
        ascending order by time, and the time intervals are assumed to be regular.
    phi : array_like
        Array of length p+1 specifying the parameters of the AR(p) model. The
        parameters are in ascending order by increasing time lag, and the last
        element is the parameter corresponding to the innovation term eps.
    eps : array_like
        Optional perturbation field for the AR(p) process. The shape of eps is
        expected to be x.shape[1:]. If eps is None, the innovation term is not
        added.

    """
    if x.shape[0] < len(phi) - 1:
        raise ValueError("dimension mismatch between x and phi: x.shape[0]=%d, len(phi)=%d" % (x.shape[0], len(phi)))

    if len(x.shape) == 1:
        x_simple_shape = True
        x = x[:, np.newaxis]
    else:
        x_simple_shape = False

    if eps is not None and eps.shape != x.shape[1:]:
        raise ValueError("dimension mismatch between x and eps: x.shape=%s, eps.shape[1:]=%s" % (str(x.shape), str(eps.shape[1:])))

    x_new = 0.0

    p = len(phi) - 1

    for i in range(p):
        x_new += phi[i] * x[-(i+1), :]

    if eps is not None:
        x_new += phi[-1] * eps

    if x_simple_shape:
        return np.hstack([x[1:], [x_new]])
    else:
        return np.concatenate([x[1:, :], x_new[np.newaxis, :]])


def iterate_var_model(x, phi, eps=None):
    """Apply a VAR(p) model

    :math:`\mathbf{x}_{k+1}=\mathbf{\Phi}_1\mathbf{x}_k+\mathbf{\Phi}_2
    \mathbf{x}_{k-1}+\dots+\mathbf{\Phi}_p\mathbf{x}_{k-p}+
    \mathbf{\Phi}_{p+1}\mathbf{\epsilon}`

    to a q-variate time series :math:`\mathbf{x}_k`.

    Parameters
    ----------
    x : array_like
        Array of shape (q,n,...), n>=p, containing a q-variate time series of a
        input variable x with length n=p. The elements of x along the second
        dimension are assumed to be in ascending order by time, and the time
        intervals are assumed to be regular.
    phi : list
        List of parameter matrices :math:`\mathbf{\Phi}_1,\mathbf{\Phi}_2,\dots,
        \mathbf{\Phi}_{p+1}`.
    eps : array_like
        The innovation term. The length is expected to be x.shape[0].
    """
    if x.shape[1] < len(phi) - 1:
        raise ValueError("dimension mismatch between x and phi: x.shape[1]=%d, len(phi)=%d" % (x.shape[1], len(phi)))

    phi_shape = phi[0].shape
    for i in range(1, len(phi)):
        if phi[i].shape != phi_shape:
            raise ValueError("dimension mismatch between parameter matrices phi")

    if len(x.shape) == 2:
        x_simple_shape = True
        x = x[:, :, np.newaxis]
    else:
        x_simple_shape = False

    x_new = np.zeros(np.hstack([[x.shape[0]], x.shape[2:]]))

    p = len(phi) - 1
    q = phi_shape[0]

    if x_simple_shape:
        for l in range(p):
            x_new += np.dot(phi[l], x[:, -(l+1), :])
    else:
        for l in range(p):
            for i in range(q):
                for j in range(q):
                    x_new[i] += phi[l][i, j] * x[j, -(l+1), :]

    if eps is not None:
        x_new += np.dot(np.dot(phi[-1], phi[-1]), eps)

    if x_simple_shape:
        return np.hstack([x[:, 1:, 0], x_new])
    else:
        return np.concatenate([x[:, 1:, :], x_new[:, np.newaxis, :]], axis=1)


def test_ar_stationarity(phi):
    """Test stationarity of an AR(p) process. That is, test that the roots of
    the equation :math:`x^p-\phi_1*x^{p-1}-\dots-\phi_p` lie inside the unit
    circle.

    Parameters
    ----------
    phi : list
        List of AR(p) parameters :math:`\phi_1,\phi_2,\dots,\phi_p`.

    Returns
    -------
    out : bool
        True/False if the process is/is not stationary.
    """
    r = np.array([np.abs(r_) for r_ in np.roots([1.0 if i == 0 else -phi[i] \
                 for i in range(len(phi))])])

    return False if np.any(r >= 1) else True


def test_var_stationarity(phi):
    """Test stationarity of an AR(p) process. That is, test that the moduli of
    the eigenvalues of the companion matrix lie inside the unit circle.

    Parameters
    ----------
    phi : list
        List of VAR(p) parameter matrices :math:`\mathbf{\Phi}_1,\mathbf{\Phi}_2,
        \dots,\mathbf{\Phi}_p`.

    Returns
    -------
    out : bool
        True/False if the process is/is not stationary.
    """
    q = phi[0].shape
    for i in range(1, len(phi)):
        if phi[i].shape != q:
            raise ValueError("dimension mismatch between parameter matrices phi")

    p = len(phi)
    q = phi[0].shape[0]

    M = np.zeros((p*q, p*q))

    for i in range(p):
        M[0:q, i*q:(i+1)*q] = phi[i]
    for i in range(1, p):
        M[i*q:(i+1)*q, (i-1)*q:i*q] = np.eye(q, q)
    r = np.linalg.eig(M)[0]

    return False if np.any(np.abs(r) >= 1) else True
