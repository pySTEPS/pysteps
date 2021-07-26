# -*- coding: utf-8 -*-
"""
pysteps.cascade.decomposition
=============================

Methods for decomposing two-dimensional fields into multiple spatial scales and
recomposing the individual scales to obtain the original field.

The methods in this module implement the following interface::

    decomposition_xxx(field, bp_filter, **kwargs)
    recompose_xxx(decomp, **kwargs)

where field is the input field and bp_filter is a dictionary returned by a
filter method implemented in :py:mod:`pysteps.cascade.bandpass_filters`. The
decomp argument is a decomposition obtained by calling decomposition_xxx.
Optional parameters can be passed in
the keyword arguments. The output of each method is a dictionary with the
following key-value pairs:

+-------------------+----------------------------------------------------------+
|        Key        |                      Value                               |
+===================+==========================================================+
|  cascade_levels   | three-dimensional array of shape (k,m,n), where k is the |
|                   | number of cascade levels and the input fields have shape |
|                   | (m,n)                                                    |
|                   | if domain is "spectral" and compact output is requested  |
|                   | (see the table below), cascade_levels contains a list of |
|                   | one-dimensional arrays                                   |
+-------------------+----------------------------------------------------------+
|  domain           | domain of the cascade decomposition: "spatial" or        |
|                   | "spectral"                                               |
+-------------------+----------------------------------------------------------+
|  normalized       | are the cascade levels normalized: True or False         |
+-------------------+----------------------------------------------------------+

The following key-value pairs are optional. They are included in the output if
``kwargs`` contains the "compute_stats" key with value set to True:

+-------------------+----------------------------------------------------------+
|        Key        |                      Value                               |
+===================+==========================================================+
|  means            | list of mean values for each cascade level               |
+-------------------+----------------------------------------------------------+
|  stds             | list of standard deviations for each cascade level       |
+-------------------+----------------------------------------------------------+

The following key-value pairs are included in the output if ``kwargs`` contains
the key "output_domain" with value set to "spectral":

+-------------------+----------------------------------------------------------+
|        Key        |                      Value                               |
+===================+==========================================================+
|  compact_output   | True or False. If set to True, only the parts of the     |
|                   | Fourier spectrum with non-negligible filter weights are  |
|                   | stored.                                                  |
+-------------------+----------------------------------------------------------+
|  weight_masks     | Applicable if compact_output is True. Contains a list of |
|                   | masks, where a True value indicates that the             |
|                   | corresponding Fourier wavenumber is included in the      |
|                   | decomposition                                            |
+-------------------+----------------------------------------------------------+


Available methods
-----------------

.. autosummary::
    :toctree: ../generated/

    decomposition_fft
    recompose_fft
"""

import numpy as np
from pysteps import utils


def decomposition_fft(field, bp_filter, **kwargs):
    """Decompose a two-dimensional input field into multiple spatial scales by
    using the Fast Fourier Transform (FFT) and a set of bandpass filters.

    Parameters
    ----------
    field: array_like
        Two-dimensional array containing the input field. All values are
        required to be finite.
    bp_filter: dict
        A filter returned by a method implemented in
        :py:mod:`pysteps.cascade.bandpass_filters`.

    Other Parameters
    ----------------
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy". This option is not used if input_domain and
        output_domain are both set to "spectral".
    normalize: bool
        If True, normalize the cascade levels to zero mean and unit variance.
        Requires that compute_stats is True. Implies that compute_stats is True.
        Defaults to False.
    mask: array_like
        Optional mask to use for computing the statistics for the cascade
        levels. Pixels with mask==False are excluded from the computations.
        This option is not used if output domain is "spectral".
    input_domain: {"spatial", "spectral"}
        The domain of the input field. If "spectral", the input is assumed to
        be in the spectral domain. Defaults to "spatial".
    output_domain: {"spatial", "spectral"}
        If "spatial", the output cascade levels are transformed back to the
        spatial domain by using the inverse FFT. If "spectral", the cascade is
        kept in the spectral domain. Defaults to "spatial".
    compute_stats: bool
        If True, the output dictionary contains the keys "means" and "stds"
        for the mean and standard deviation of each output cascade level.
        Defaults to False.
    compact_output: bool
        Applicable if output_domain is "spectral". If set to True, only the
        parts of the Fourier spectrum with non-negligible filter weights are
        stored. Defaults to False.

    Returns
    -------
    out: ndarray
        A dictionary described in the module documentation.
        The number of cascade levels is determined from the filter
        (see :py:mod:`pysteps.cascade.bandpass_filters`).

    """
    fft = kwargs.get("fft_method", "numpy")
    if isinstance(fft, str):
        fft = utils.get_method(fft, shape=field.shape)
    normalize = kwargs.get("normalize", False)
    mask = kwargs.get("mask", None)
    input_domain = kwargs.get("input_domain", "spatial")
    output_domain = kwargs.get("output_domain", "spatial")
    compute_stats = kwargs.get("compute_stats", True)
    compact_output = kwargs.get("compact_output", False)

    if normalize and not compute_stats:
        compute_stats = True

    if len(field.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if mask is not None and mask.shape != field.shape:
        raise ValueError(
            "Dimension mismatch between field and mask:"
            + "field.shape="
            + str(field.shape)
            + ",mask.shape"
            + str(mask.shape)
        )

    if field.shape[0] != bp_filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between field and bp_filter: "
            + "field.shape[0]=%d , " % field.shape[0]
            + "bp_filter['weights_2d'].shape[1]"
            "=%d" % bp_filter["weights_2d"].shape[1]
        )

    if (
        input_domain == "spatial"
        and int(field.shape[1] / 2) + 1 != bp_filter["weights_2d"].shape[2]
    ):
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "int(field.shape[1]/2)+1=%d , " % (int(field.shape[1] / 2) + 1)
            + "bp_filter['weights_2d'].shape[2]"
            "=%d" % bp_filter["weights_2d"].shape[2]
        )

    if (
        input_domain == "spectral"
        and field.shape[1] != bp_filter["weights_2d"].shape[2]
    ):
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "field.shape[1]=%d , " % (field.shape[1] + 1)
            + "bp_filter['weights_2d'].shape[2]"
            "=%d" % bp_filter["weights_2d"].shape[2]
        )

    if output_domain != "spectral":
        compact_output = False

    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    result = {}
    means = []
    stds = []

    if input_domain == "spatial":
        field_fft = fft.rfft2(field)
    else:
        field_fft = field
    if output_domain == "spectral" and compact_output:
        weight_masks = []
    field_decomp = []

    for k in range(len(bp_filter["weights_1d"])):
        field_ = field_fft * bp_filter["weights_2d"][k, :, :]

        if output_domain == "spatial" or (compute_stats and mask is not None):
            field__ = fft.irfft2(field_)
        else:
            field__ = field_

        if compute_stats:
            if output_domain == "spatial" or (compute_stats and mask is not None):
                if mask is not None:
                    masked_field = field__[mask]
                else:
                    masked_field = field__
                mean = np.mean(masked_field)
                std = np.std(masked_field)
            else:
                mean = utils.spectral.mean(field_, bp_filter["shape"])
                std = utils.spectral.std(field_, bp_filter["shape"])

            means.append(mean)
            stds.append(std)

        if output_domain == "spatial":
            field_ = field__
        if normalize:
            field_ = (field_ - mean) / std
        if output_domain == "spectral" and compact_output:
            weight_mask = bp_filter["weights_2d"][k, :, :] > 1e-12
            field_ = field_[weight_mask]
            weight_masks.append(weight_mask)
        field_decomp.append(field_)

    result["domain"] = output_domain
    result["normalized"] = normalize
    result["compact_output"] = compact_output

    if output_domain == "spatial" or not compact_output:
        field_decomp = np.stack(field_decomp)

    result["cascade_levels"] = field_decomp
    if output_domain == "spectral" and compact_output:
        result["weight_masks"] = np.stack(weight_masks)

    if compute_stats:
        result["means"] = means
        result["stds"] = stds

    return result


def recompose_fft(decomp, **kwargs):
    """Recompose a cascade obtained with decomposition_fft by inverting the
    normalization and summing the cascade levels.

    Parameters
    ----------
    decomp: dict
        A cascade decomposition returned by decomposition_fft.

    Returns
    -------
    out: numpy.ndarray
        A two-dimensional array containing the recomposed cascade.
    """
    levels = decomp["cascade_levels"]
    if decomp["normalized"]:
        mu = decomp["means"]
        sigma = decomp["stds"]

    if not decomp["normalized"] and not (
        decomp["domain"] == "spectral" and decomp["compact_output"]
    ):
        return np.sum(levels, axis=0)
    else:
        if decomp["compact_output"]:
            weight_masks = decomp["weight_masks"]
            result = np.zeros(weight_masks.shape[1:], dtype=complex)

            for i in range(len(levels)):
                if decomp["normalized"]:
                    result[weight_masks[i]] += levels[i] * sigma[i] + mu[i]
                else:
                    result[weight_masks[i]] += levels[i]
        else:
            result = [levels[i] * sigma[i] + mu[i] for i in range(len(levels))]
            result = np.sum(np.stack(result), axis=0)

        return result
