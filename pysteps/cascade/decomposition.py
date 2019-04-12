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


def decomposition_fft(X, filter, **kwargs):
    """Decompose a 2d input field into multiple spatial scales by using the Fast
    Fourier Transform (FFT) and a bandpass filter.

    Parameters
    ----------
    X : array_like
        Two-dimensional array containing the input field. All values are
        required to be finite.
    filter : dict
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
    input_domain : {"spatial", "spectral"}
        The domain of the inputs. If "spectral", the FFT is assumed to be
        applied to the inputs.
    output_domain : {"spatial", "spectral"}
        If "spatial", the output cascade levels are transformed back to the
        spatial domain by using the inverse FFT. If "spectral", the cascade is
        kept in the spectral domain.
    compute_stats : bool
        If True, the output dictionary contains the keys "means" and "stds"
        for the mean and standard deviation of each output cascade level.

    Returns
    -------
    out : ndarray
        A dictionary described in the module documentation.
        The number of cascade levels is determined from the filter
        (see :py:mod:`pysteps.cascade.bandpass_filters`).

    """
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft = utils.get_method(fft, shape=X.shape)
    input_domain = kwargs.get("input_domain", "spatial")
    output_domain = kwargs.get("output_domain", "spatial")
    compute_stats = kwargs.get("compute_stats", True)

    MASK = kwargs.get("MASK", None)

    if len(X.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if MASK is not None and MASK.shape != X.shape:
        raise ValueError("Dimension mismatch between X and MASK:"
                         + "X.shape=" + str(X.shape)
                         + ",MASK.shape" + str(MASK.shape))

    if X.shape[0] != filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between X and filter: "
            + "X.shape[0]=%d , " % X.shape[0]
            + "filter['weights_2d'].shape[1]"
              "=%d" % filter["weights_2d"].shape[1])

    if int(X.shape[1] / 2) + 1 != filter["weights_2d"].shape[2]:
        raise ValueError(
            "Dimension mismatch between X and filter: "
            "int(X.shape[1]/2)+1=%d , " % (int(X.shape[1] / 2) + 1)
            + "filter['weights_2d'].shape[2]"
              "=%d" % filter["weights_2d"].shape[2])

    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    result = {}
    means = []
    stds = []

    if input_domain == "spatial":
        F = fft.rfft2(X)
    else:
        F = X
    X_decomp = []
    for k in range(len(filter["weights_1d"])):
        W_k = filter["weights_2d"][k, :, :]
        X_ = F * W_k
        if output_domain == "spatial" or compute_stats:
            X__ = fft.irfft2(X_)
        if output_domain == "spatial":
            X_decomp.append(X__)
        else:
            X_decomp.append(X_)

        if output_domain == "spatial":
            if MASK is not None:
                X__ = X_[MASK]

        if compute_stats:
            means.append(np.mean(X__))
            stds.append(np.std(X__))

    result["cascade_levels"] = np.stack(X_decomp)
    if compute_stats:
        result["means"] = means
        result["stds"] = stds

    return result
