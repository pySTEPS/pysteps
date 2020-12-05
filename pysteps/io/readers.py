# -*- coding: utf-8 -*-
"""
pysteps.io.readers
==================

Module with the reader functions.

.. autosummary::
    :toctree: ../generated/

    read_timeseries
"""

import numpy as np


def read_timeseries(inputfns, importer, **kwargs):
    """Read a time series of input files using the methods implemented in the
    :py:mod:`pysteps.io.importers` module and stack them into a 3d array of
    shape (num_timesteps, height, width).

    Parameters
    ----------
    inputfns: tuple
        Input files returned by a function implemented in the
        :py:mod:`pysteps.io.archive` module.
    importer: function
        A function implemented in the :py:mod:`pysteps.io.importers` module.
    kwargs: dict
        Optional keyword arguments for the importer.

    Returns
    -------
    out: tuple
        A three-element tuple containing the read data and quality rasters and
        associated metadata. If an input file name is None, the corresponding
        precipitation and quality fields are filled with nan values. If all
        input file names are None or if the length of the file name list is
        zero, a three-element tuple containing None values is returned.

    """

    # check for missing data
    precip_ref = None
    if all(ifn is None for ifn in inputfns):
        return None, None, None
    else:
        if len(inputfns[0]) == 0:
            return None, None, None
        for ifn in inputfns[0]:
            if ifn is not None:
                precip_ref, quality_ref, metadata = importer(ifn, **kwargs)
                break

    if precip_ref is None:
        return None, None, None

    precip = []
    quality = []
    timestamps = []
    for i, ifn in enumerate(inputfns[0]):
        if ifn is not None:
            precip_, quality_, _ = importer(ifn, **kwargs)
            precip.append(precip_)
            quality.append(quality_)
            timestamps.append(inputfns[1][i])
        else:
            precip.append(precip_ref * np.nan)
            if quality_ref is not None:
                quality.append(quality_ref * np.nan)
            else:
                quality.append(None)
            timestamps.append(inputfns[1][i])

    # Replace this with stack?
    precip = np.concatenate([precip_[None, :, :] for precip_ in precip])
    # TODO: Q should be organized as R, but this is not trivial as Q_ can be also None or a scalar
    metadata["timestamps"] = np.array(timestamps)

    return precip, quality, metadata
