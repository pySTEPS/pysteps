"""
pysteps.cascade.bandpass_filters
================================

Bandpass filters for separating different spatial scales from two-dimensional
images in the frequency domain.

The methods in this module implement the following interface::

    filter_xxx(shape, n, optional arguments)

where shape is the shape of the input field, respectively, and n is the number
of frequency bands to use.

The output of each filter function is a dictionary containing the following
key-value pairs:

.. tabularcolumns:: |p{1.8cm}|L|

+-----------------+-----------------------------------------------------------+
|       Key       |                Value                                      |
+=================+===========================================================+
| weights_1d      | 2d array of shape (n, r) containing 1d filter weights for |
|                 | each frequency band k=1,2,...,n                           |
+-----------------+-----------------------------------------------------------+
| weights_2d      | 3d array of shape (n, M, int(N/2)+1) containing the 2d    |
|                 | filter weights for each frequency band k=1,2,...,n        |
+-----------------+-----------------------------------------------------------+
| central_freqs   | 1d array of shape n containing the central frequencies of |
|                 | the filters                                               |
+-----------------+-----------------------------------------------------------+

where r = int(max(N, M)/2)+1

By default, the filter weights are normalized so that for any Fourier wavenumber
they sum to one.

Available filters
-----------------

.. autosummary::
    :toctree: ../generated/

    filter_uniform
    filter_gaussian
"""

import numpy as np

def filter_uniform(shape, n):
    """A dummy filter with one frequency band covering the whole domain. The
    weights are set to one.

    Parameters
    ----------
    shape : int or tuple
        The dimensions (height, width) of the input field. If shape is an int,
        the domain is assumed to have square shape.
    n : int
        Not used. Needed for compatibility with the filter interface.

    """
    result = {}

    try:
        M,N = shape
    except TypeError:
        M,N = (shape, shape)

    r_max = int(max(N, M)/2)+1

    result["weights_1d"]    = np.ones((1, r_max))
    result["weights_2d"]    = np.ones((1, M, int(N/2)+1))
    result["central_freqs"] = None
    result["central_wavenumbers"] = None

    return result

def filter_gaussian(shape, n, l_0=3, gauss_scale=0.5, gauss_scale_0=0.5, d=1.0,
                    normalize=True):
    """Implements a set of Gaussian bandpass filters in logarithmic frequency
    scale.

    Parameters
    ----------
    shape : int or tuple
        The dimensions (height, width) of the input field. If shape is an int,
        the domain is assumed to have square shape.
    n : int
        The number of frequency bands to use. Must be greater than 2.
    l_0 : int
        Central frequency of the second band (the first band is always centered
        at zero).
    gauss_scale : float
        Optional scaling prameter. Proportional to the standard deviation of the
        Gaussian weight functions.
    gauss_scale_0 : float
        Optional scaling parameter for the Gaussian function corresponding to
        the first frequency band.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    normalize : bool
        If True, normalize the weights so that for any given wavenumber they sum
        to one.

    Returns
    -------
    out : dict
        A dictionary containing the bandpass filters corresponding to the
        specified frequency bands.

    References
    ----------
    :cite:`PCH2018`

    """
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        M,N = shape
    except TypeError:
        M,N = (shape, shape)

    rx = np.s_[:int(N/2)+1]

    if M % 2 == 1:
        ry = np.s_[-int(M/2):int(M/2)+1]
    else:
        ry = np.s_[-int(M/2):int(M/2)]

    Y,X = np.ogrid[ry, rx]
    dy = int(M/2) if M % 2 == 0 else int(M/2)+1
    R = np.roll(np.sqrt(X*X + Y*Y), dy, axis=0)

    L = max(N, M)
    r_max = int(L/2)+1
    r = np.arange(r_max)

    wfs,cwn = _gaussweights_1d(L, n, l_0=l_0, gauss_scale=gauss_scale,
                               gauss_scale_0=gauss_scale_0)

    w = np.empty((n, r_max))
    W = np.empty((n, M, int(N/2)+1))

    for i,wf in enumerate(wfs):
        w[i, :] = wf(r)
        W[i, :, :] = wf(R)

    if normalize:
        w_sum = np.sum(w, axis=0)
        W_sum = np.sum(W, axis=0)
        for k in range(W.shape[0]):
            w[k, :]    /= w_sum
            W[k, :, :] /= W_sum

    result = {}
    result["weights_1d"] = w
    result["weights_2d"] = W
    
    cwn = np.array(cwn)
    result["central_wavenumbers"] = cwn
    
    # Compute frequencies
    central_freqs = 1.0*cwn/L
    central_freqs[0] = 1.0/L
    central_freqs[-1] = 0.5 # Nyquist freq
    central_freqs = 1.0*d*central_freqs
    result["central_freqs"] = central_freqs

    return result

def _gaussweights_1d(l, n, l_0=3, gauss_scale=0.5, gauss_scale_0=0.5):
    e = pow(0.5*l/l_0, 1.0/(n-2))
    r = [(l_0*pow(e, k-1), l_0*pow(e, k)) for k in range(1, n-1)]

    f = lambda x,s: np.exp(-x**2.0 / (2.0*s**2.0))
    def log_e(x):
        if len(np.shape(x)) > 0:
            res = np.empty(x.shape)
            res[x == 0] = 0.0
            res[x > 0] = np.log(x[x > 0]) / np.log(e)
        else:
            if x == 0.0:
                res = 0.0
            else:
                res = np.log(x) / np.log(e)

        return res

    class gaussfunc:

        def __init__(self, c, s):
            self.c = c
            self.s = s

        def __call__(self, x):
            return f(log_e(x) - self.c, self.s)

    weight_funcs  = []
    central_wavenumbers = [0.0]

    s = gauss_scale
    weight_funcs.append(gaussfunc(0.0, gauss_scale_0))

    for i,ri in enumerate(r):
        rc = log_e(ri[0])
        weight_funcs.append(gaussfunc(rc, s))
        central_wavenumbers.append(ri[0])

    gf = gaussfunc(log_e(l/2), s)
    def g(x):
        res = np.ones(x.shape)
        mask = x <= l/2
        res[mask] = gf(x[mask])

        return res

    weight_funcs.append(g)
    central_wavenumbers.append(l/2)

    return weight_funcs, central_wavenumbers
