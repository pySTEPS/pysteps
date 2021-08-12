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
import xarray as xr

from pysteps.decorators import _xarray2legacy


def read_timeseries(input_fns, importer, **kwargs):
    """Read a time series of input files using the methods implemented in the
    :py:mod:`pysteps.io.importers` module and concatenate them into a 3d array
    of shape (t, y, x).

    Parameters
    ----------
    input_fns: tuple
        Input files returned by a function implemented in the
        :py:mod:`pysteps.io.archive` module.
    importer: function
        A function implemented in the :py:mod:`pysteps.io.importers` module.
    kwargs: dict
        Optional keyword arguments for the importer.

    Returns
    -------
    out: xr.DataArray
        A xarray DataArray containing the data rasters and
        associated metadata. If an input file name is None, the corresponding
        fields are filled with nan values.
        If all input file names are None or if the length of the file name list
        is zero, None is returned.
    """
    legacy = kwargs.get("legacy", False)
    kwargs["legacy"] = False
    # check for missing data
    precip_ref = None
    if all(ifn is None for ifn in input_fns):
        return None
    else:
        if len(input_fns[0]) == 0:
            return None
        for ifn in input_fns[0]:
            if ifn is not None:
                precip_ref = importer(ifn, **kwargs)
                break

    if precip_ref is None:
        return None

    precip = []
    timestamps = []
    for i, ifn in enumerate(input_fns[0]):
        if ifn is not None:
            precip_ = importer(ifn, **kwargs)
            precip.append(precip_)
            timestamps.append(input_fns[1][i])
        else:
            precip.append(xr.full_like(precip_ref, np.nan))
            timestamps.append(input_fns[1][i])

    precip = xr.concat(precip, "t")
    precip = precip.assign_coords({"t": timestamps})

    if legacy:
        return _xarray2legacy(precip)

    return precip
