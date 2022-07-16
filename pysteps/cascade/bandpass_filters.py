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
| shape           | the shape of the input field in the spatial domain        |
+-----------------+-----------------------------------------------------------+

where r = int(max(N, M)/2)+1

By default, the filter weights are normalized so that for any Fourier
wavenumber they sum to one.

Available filters
-----------------

.. autosummary::
    :toctree: ../generated/

    filter_uniform
    filter_gaussian
"""

import numpy as np


def filter_uniform(shape, n):
    """
    A dummy filter with one frequency band covering the whole domain. The
    weights are set to one.

    Parameters
    ----------
    shape: int or tuple
        The dimensions (height, width) of the input field. If shape is an int,
        the domain is assumed to have square shape.
    n: int
        Not used. Needed for compatibility with the filter interface.

    Returns
    -------
    out: dict
        A dictionary containing the filter.
    """
    del n  # Unused

    out = {}

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    r_max = int(max(width, height) / 2) + 1

    out["weights_1d"] = np.ones((1, r_max))
    out["weights_2d"] = np.ones((1, height, int(width / 2) + 1))
    out["central_freqs"] = None
    out["central_wavenumbers"] = None
    out["shape"] = shape

    return out


def filter_gaussian(
    shape,
    n,
    gauss_scale=0.5,
    d=1.0,
    normalize=True,
    return_weight_funcs=False,
    include_mean=True,
):
    """
    Implements a set of Gaussian bandpass filters in logarithmic frequency
    scale.

    Parameters
    ----------
    shape: int or tuple
        The dimensions (height, width) of the input field. If shape is an int,
        the domain is assumed to have square shape.
    n: int
        The number of frequency bands to use. Must be greater than 2.
    gauss_scale: float
        Optional scaling parameter. Proportional to the standard deviation of
        the Gaussian weight functions.
    d: scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    normalize: bool
        If True, normalize the weights so that for any given wavenumber
        they sum to one.
    return_weight_funcs: bool
        If True, add callable weight functions to the output dictionary with
        the key 'weight_funcs'.
    include_mean: bool
        If True, include the first Fourier wavenumber (corresponding to the
        field mean) to the first filter.

    Returns
    -------
    out: dict
        A dictionary containing the bandpass filters corresponding to the
        specified frequency bands.

    References
    ----------
    :cite:`PCH2018`

    """
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    max_length = max(width, height)

    rx = np.s_[: int(width / 2) + 1]

    if (height % 2) == 1:
        ry = np.s_[-int(height / 2) : int(height / 2) + 1]
    else:
        ry = np.s_[-int(height / 2) : int(height / 2)]

    y_grid, x_grid = np.ogrid[ry, rx]
    dy = int(height / 2) if height % 2 == 0 else int(height / 2) + 1

    r_2d = np.roll(np.sqrt(x_grid * x_grid + y_grid * y_grid), dy, axis=0)

    r_max = int(max_length / 2) + 1
    r_1d = np.arange(r_max)

    wfs, central_wavenumbers = _gaussweights_1d(
        max_length,
        n,
        gauss_scale=gauss_scale,
    )

    weights_1d = np.empty((n, r_max))
    weights_2d = np.empty((n, height, int(width / 2) + 1))

    for i, wf in enumerate(wfs):
        weights_1d[i, :] = wf(r_1d)
        weights_2d[i, :, :] = wf(r_2d)

    if normalize:
        weights_1d_sum = np.sum(weights_1d, axis=0)
        weights_2d_sum = np.sum(weights_2d, axis=0)
        for k in range(weights_2d.shape[0]):
            weights_1d[k, :] /= weights_1d_sum
            weights_2d[k, :, :] /= weights_2d_sum

    for i in range(len(wfs)):
        if i == 0 and include_mean:
            weights_1d[i, 0] = 1.0
            weights_2d[i, 0, 0] = 1.0
        else:
            weights_1d[i, 0] = 0.0
            weights_2d[i, 0, 0] = 0.0

    out = {"weights_1d": weights_1d, "weights_2d": weights_2d}
    out["shape"] = shape

    central_wavenumbers = np.array(central_wavenumbers)
    out["central_wavenumbers"] = central_wavenumbers

    # Compute frequencies
    central_freqs = 1.0 * central_wavenumbers / max_length
    central_freqs[0] = 1.0 / max_length
    central_freqs[-1] = 0.5  # Nyquist freq
    central_freqs = 1.0 * d * central_freqs
    out["central_freqs"] = central_freqs

    if return_weight_funcs:
        out["weight_funcs"] = wfs

    return out


def _gaussweights_1d(l, n, gauss_scale=0.5):
    q = pow(0.5 * l, 1.0 / n)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n + 1)]
    r = [0.5 * (r_[0] + r_[1]) for r_ in r]

    def log_e(x):
        if len(np.shape(x)) > 0:
            res = np.empty(x.shape)
            res[x == 0] = 0.0
            res[x > 0] = np.log(x[x > 0]) / np.log(q)
        else:
            if x == 0.0:
                res = 0.0
            else:
                res = np.log(x) / np.log(q)

        return res

    class GaussFunc:
        def __init__(self, c, s):
            self.c = c
            self.s = s

        def __call__(self, x):
            x = log_e(x) - self.c
            return np.exp(-(x**2.0) / (2.0 * self.s**2.0))

    weight_funcs = []
    central_wavenumbers = []

    for i, ri in enumerate(r):
        rc = log_e(ri)
        weight_funcs.append(GaussFunc(rc, gauss_scale))
        central_wavenumbers.append(ri)

    return weight_funcs, central_wavenumbers
