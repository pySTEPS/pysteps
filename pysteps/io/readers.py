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


def read_timeseries(inputfns, importer, **kwargs) -> xr.Dataset | None:
    """
    Read a time series of input files using the methods implemented in the
    :py:mod:`pysteps.io.importers` module and stack them into a 3d xarray
    dataset of shape (num_timesteps, height, width).

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
    out: Dataset
        A dataset containing the read data and quality rasters and
        associated metadata. If an input file name is None, the corresponding
        precipitation and quality fields are filled with nan values. If all
        input file names are None or if the length of the file name list is
        zero, None is returned.

    """

    # check for missing data
    dataset_ref = None
    if all(ifn is None for ifn in inputfns):
        return None
    else:
        if len(inputfns[0]) == 0:
            return None
        for ifn in inputfns[0]:
            if ifn is not None:
                dataset_ref = importer(ifn, **kwargs)
                break

    if dataset_ref is None:
        return None

    startdate = min(inputfns[1])

    datasets = []
    for i, ifn in enumerate(inputfns[0]):
        if ifn is not None:
            dataset_ = importer(ifn, **kwargs)
        else:
            dataset_ = dataset_ref * np.nan
        dataset_ = dataset_.expand_dims(dim="time", axis=0)
        dataset_ = dataset_.assign_coords(
            time=(
                "time",
                [inputfns[1][i]],
                {
                    "long_name": "forecast time",
                    "units": f"seconds since {startdate:%Y-%m-%d %H:%M:%S}",
                },
            )
        )
        datasets.append(dataset_)

    dataset = xr.concat(datasets, dim="time")
    return dataset
