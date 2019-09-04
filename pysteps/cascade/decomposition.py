"""
pysteps.cascade.decomposition
=============================

Methods for decomposing two-dimensional images into multiple spatial scales.

The methods in this module implement the following interface::

    decomposition_xxx(X, filter, **kwargs)

where X is the input field and filter is a dictionary returned by a filter
method implemented in :py:mod:`pysteps.cascade.bandpass_filters`.
Optional parameters can be passed in
the keyword arguments. The output of each method is a dictionary with the
following key-value pairs:

+-------------------+----------------------------------------------------------+
|        Key        |                      Value                               |
+===================+==========================================================+
|  cascade_levels   | three-dimensional array of shape (k,m,n), where k is the |
|                   | number of cascade levels and the input fields have shape |
|                   | (m,n)                                                    |
+-------------------+----------------------------------------------------------+
|  means            | list of mean values for each cascade level               |
+-------------------+----------------------------------------------------------+
|  stds             | list of standard deviations for each cascade level       |
+-------------------+----------------------------------------------------------+

Available methods
-----------------

.. autosummary::
    :toctree: ../generated/

    decomposition_fft
"""

import numpy as np
from pysteps import utils


def decomposition_fft(field, bp_filter, **kwargs):
    """Decompose a 2d input field into multiple spatial scales by using the Fast
    Fourier Transform (FFT) and a bandpass filter.

    Parameters
    ----------
    field : array_like
        Two-dimensional array containing the input field. All values are
        required to be finite.
    bp_filter : dict
        A filter returned by a method implemented in
        :py:mod:`pysteps.cascade.bandpass_filters`.

    Other Parameters
    ----------------
    fft_method : str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".
    MASK : array_like
        Optional mask to use for computing the statistics for the cascade
        levels. Pixels with MASK==False are excluded from the computations.

    Returns
    -------
    out : ndarray
        A dictionary described in the module documentation.
        The number of cascade levels is determined from the filter
        (see :py:mod:`pysteps.cascade.bandpass_filters`).

    """
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft = utils.get_method(fft, shape=field.shape)

    mask = kwargs.get("MASK", None)

    if len(field.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if mask is not None and mask.shape != field.shape:
        raise ValueError("Dimension mismatch between X and MASK:"
                         + "X.shape=" + str(field.shape)
                         + ",mask.shape" + str(mask.shape))

    if field.shape[0] != bp_filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between X and filter: "
            + "X.shape[0]=%d , " % field.shape[0]
            + "filter['weights_2d'].shape[1]"
              "=%d" % bp_filter["weights_2d"].shape[1])

    if int(field.shape[1] / 2) + 1 != bp_filter["weights_2d"].shape[2]:
        raise ValueError(
            "Dimension mismatch between X and filter: "
            "int(X.shape[1]/2)+1=%d , " % (int(field.shape[1] / 2) + 1)
            + "filter['weights_2d'].shape[2]"
              "=%d" % bp_filter["weights_2d"].shape[2])

    if np.any(~np.isfinite(field)):
        raise ValueError("X contains non-finite values")

    result = {}
    means = []
    stds = []

    field_decomp = []

    for k in range(len(bp_filter["weights_1d"])):

        _decomp_field = fft.irfft2(fft.rfft2(field) * bp_filter["weights_2d"][k, :, :])

        field_decomp.append(_decomp_field)

        if mask is not None:
            _decomp_field = _decomp_field[mask]
        means.append(np.mean(_decomp_field))
        stds.append(np.std(_decomp_field))

    result["cascade_levels"] = np.stack(field_decomp)
    result["means"] = means
    result["stds"] = stds

    return result
