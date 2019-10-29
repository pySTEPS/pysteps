"""
pysteps.cascade.decomposition
=============================

Methods for decomposing two-dimensional fields into multiple spatial scales.

The methods in this module implement the following interface::

    decomposition_xxx(field, bp_filter, **kwargs)

where field is the input field and bp_filter is a dictionary returned by a filter
method implemented in :py:mod:`pysteps.cascade.bandpass_filters`.
Optional parameters can be passed in
the keyword arguments. The output of each method is a dictionary with the
following key-value pairs, where means and stds are optional:

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
    """Decompose a two-dimensional input field into multiple spatial scales by
    using the Fast Fourier Transform (FFT) and a set of bandpass filters.

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
    input_domain : {"spatial", "spectral"}
        The domain of the input field. If "spectral", the input is assumed to
        be in the spectral domain.
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
    if isinstance(fft, str):
        fft = utils.get_method(fft, shape=field.shape)
    mask = kwargs.get("MASK", None)
    input_domain = kwargs.get("input_domain", "spatial")
    output_domain = kwargs.get("output_domain", "spatial")
    compute_stats = kwargs.get("compute_stats", True)

    if len(field.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if mask is not None and mask.shape != field.shape:
        raise ValueError("Dimension mismatch between field and MASK:"
                         + "field.shape=" + str(field.shape)
                         + ",mask.shape" + str(mask.shape))

    if field.shape[0] != bp_filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between field and bp_filter: "
            + "field.shape[0]=%d , " % field.shape[0]
            + "bp_filter['weights_2d'].shape[1]"
              "=%d" % bp_filter["weights_2d"].shape[1])

    if input_domain == "spatial" and \
       int(field.shape[1] / 2) + 1 != bp_filter["weights_2d"].shape[2]:
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "int(field.shape[1]/2)+1=%d , " % (int(field.shape[1] / 2) + 1)
            + "bp_filter['weights_2d'].shape[2]"
              "=%d" % bp_filter["weights_2d"].shape[2])

    if input_domain == "spectral" and \
       field.shape[1] != bp_filter["weights_2d"].shape[2]:
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "field.shape[1]=%d , " % (field.shape[1] + 1)
            + "bp_filter['weights_2d'].shape[2]"
              "=%d" % bp_filter["weights_2d"].shape[2])

    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    result = {}
    means = []
    stds = []

    if input_domain == "spatial":
        field_fft = fft.rfft2(field)
    else:
        field_fft = field
    field_decomp = []

    for k in range(len(bp_filter["weights_1d"])):
        field_ = field_fft * bp_filter["weights_2d"][k, :, :]
        if output_domain == "spatial" or compute_stats:
            field__ = fft.irfft2(field_)
        if output_domain == "spatial":
            field_decomp.append(field__)
        else:
            field_decomp.append(field_)

        if output_domain == "spatial" and mask is not None:
            field__ = field_[mask]

        if compute_stats:
            means.append(np.mean(field__))
            stds.append(np.std(field__))

    result["cascade_levels"] = np.stack(field_decomp)

    if compute_stats:
        result["means"] = means
        result["stds"] = stds

    return result
