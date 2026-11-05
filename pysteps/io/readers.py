# -*- coding: utf-8 -*-
"""
pysteps.io.readers
==================

Module with the reader functions.

.. autosummary::
    :toctree: ../generated/

    read_timeseries
"""

import warnings

import numpy as np
import xarray as xr

from pysteps.dataset_metadata import (
    set_time_metadata,
    temporal_method,
    time_bounds,
    utc_datetime,
)


def read_timeseries(inputfns, importer, timestep=None, **kwargs) -> xr.Dataset | None:
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
    timestep: int, optional
        The timestep in seconds, this value is optional if more than 1 inputfns
        are given.
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
    sorted_dates = sorted(inputfns[1])
    timestep_dates = None
    if len(sorted_dates) > 1:
        timestep_dates = float((sorted_dates[1] - sorted_dates[0]).total_seconds())

    if timestep is None and timestep_dates is None:
        raise ValueError("either provide a timestep or provide more than one inputfn")
    if timestep is None:
        timestep = timestep_dates
    if timestep_dates is not None and timestep != timestep_dates:
        # XR: This should be an error, but some test fail on this.
        warnings.warn(
            "Supplied timestep does not match actual timestep spacing in input data, "
            + "using actual spacing as timestep."
        )
        timestep = timestep_dates
    for i in range(len(sorted_dates) - 1):
        if not np.isclose(
            (sorted_dates[i + 1] - sorted_dates[i]).total_seconds(),
            timestep,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise ValueError("supplied dates are not evenly spaced")

    if "time" in dataset_ref.dims:
        return _read_cf_frames(inputfns, importer, dataset_ref, timestep, **kwargs)

    datasets = []
    for i, ifn in enumerate(inputfns[0]):
        if ifn is not None:
            dataset_ = importer(ifn, **kwargs)
            precip_var = dataset_.attrs["precip_var"]
            precip_var_ref = dataset_ref.attrs["precip_var"]

            # The initial metadata setting reads the oldest time step only (see previous importer call).
            # In case this has rain over the entire domain, the no precip threshold
            # might be higher than the actual zero value.
            # Here we check all time steps and use the lowest value overall.
            if (
                "threshold" in dataset_ref[precip_var_ref].attrs
                and "threshold" in dataset_[precip_var].attrs
            ):
                dataset_ref[precip_var_ref].attrs["threshold"] = min(
                    dataset_ref[precip_var_ref].attrs["threshold"],
                    dataset_[precip_var].attrs["threshold"],
                )
        else:
            dataset_ = dataset_ref * np.nan
        dataset_ = dataset_.expand_dims(dim="time", axis=0)
        dataset_ = dataset_.assign_coords(
            time=(
                "time",
                [utc_datetime(inputfns[1][i])],
                {
                    "standard_name": "time",
                    "long_name": "valid time",
                    "stepsize": float(timestep),
                },
                {"units": f"seconds since {startdate:%Y-%m-%d %H:%M:%S}"},
            )
        )
        variable = dataset_.attrs["precip_var"]
        if temporal_method(dataset_, variable) != "point":
            bounds = time_bounds(dataset_, variable, legacy=True)
            if bounds is None:
                raise ValueError("Legacy interval input needs an explicit duration")
            dataset_ = set_time_metadata(dataset_, variable, bounds)
        datasets.append(dataset_)

    dataset = xr.concat(datasets, dim="time")
    variable = dataset.attrs["precip_var"]
    if temporal_method(dataset, variable) != "point":
        dataset[variable].attrs.pop("accutime", None)
        dataset = set_time_metadata(dataset, variable, time_bounds(dataset, variable))
        dataset.attrs["history"] = (
            dataset.attrs.get("history", "")
            + "\nLegacy duration interpreted as interval ending at archive valid time"
        ).strip()
    return dataset


def _read_cf_frames(inputfns, importer, reference, timestep, **kwargs):
    """Stack explicitly dated singleton CF files without replacing their times."""
    from pysteps.dataset_metadata import physical_variables, temporal_axis
    from pyproj import CRS

    if kwargs.get("load") is False:
        reference.close()
        raise ValueError(
            "read_timeseries needs eager CF frames; open lazy files directly"
        )
    first = next(name for name in inputfns[0] if name is not None)
    physical = physical_variables(reference)
    missing = any(name is None for name in inputfns[0])
    datasets = []
    for filename, date in zip(inputfns[0], inputfns[1]):
        date = utc_datetime(date)
        frame = (
            reference.copy(deep=False)
            if filename is None or filename == first
            else importer(filename, **kwargs)
        )
        if (
            any(temporal_axis(frame, name) != "time" for name in physical)
            or frame.sizes.get("time") != 1
        ):
            raise ValueError(
                "CF archive frames must contain one valid time on a shared time axis; import multi-time files directly"
            )
        if physical_variables(frame) != physical:
            raise ValueError("CF archive physical variables differ between files")
        if not CRS(frame.attrs["projection"]).equals(
            CRS(reference.attrs["projection"])
        ) or any(
            not np.array_equal(frame[axis], reference[axis]) for axis in ("x", "y")
        ):
            raise ValueError("CF archive grid/CRS differs between files")
        for name in physical:
            if frame[name].attrs["units"] != reference[name].attrs[
                "units"
            ] or temporal_method(frame, name) != temporal_method(reference, name):
                raise ValueError(
                    "CF archive quantity or temporal meaning differs between files"
                )
            if (
                missing
                and temporal_method(frame, name) != "point"
                and frame[name].attrs.get("accutime")
                != reference[name].attrs.get("accutime")
            ):
                raise ValueError(
                    "Padding missing CF intervals requires one known fixed duration"
                )
        if filename is None:
            delta = date - frame.time.values[0]
            for name in frame.data_vars:
                if "time" in frame[name].dims:
                    value = frame[name]
                    frame[name] = (value.dims, value.data * np.nan, value.attrs.copy())
            for name in list(frame.coords):
                if "time" in frame[name].dims and np.issubdtype(
                    frame[name].dtype, np.datetime64
                ):
                    coordinate = frame[name]
                    frame = frame.assign_coords(
                        {
                            name: (
                                coordinate.dims,
                                coordinate.values + delta,
                                coordinate.attrs,
                            )
                        }
                    )
            for name in physical:
                if (
                    temporal_method(frame, name) != "point"
                    and frame[name].attrs.get("accutime") is None
                ):
                    raise ValueError(
                        "Padding missing CF intervals requires one known fixed duration"
                    )
        elif frame.time.values[0] != date:
            raise ValueError("CF valid time conflicts with the archive date")
        for name in physical:
            for key in ("threshold", "zerovalue"):
                if frame[name].attrs.get(key) != reference[name].attrs.get(key):
                    raise ValueError(
                        f"CF archive {key} varies; provide an explicit parameter-field policy"
                    )
            frame[name].attrs.pop("accutime", None)
        frame.time.attrs.pop("stepsize", None)
        if (
            "forecast_reference_time" in frame
            and "forecast_reference_time" in reference
            and not np.array_equal(
                frame.forecast_reference_time, reference.forecast_reference_time
            )
        ):
            raise ValueError(
                "CF archive forecast origins differ; import forecasts separately"
            )
        datasets.append(frame)
    dataset = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
    )
    for name in physical:
        method = temporal_method(dataset, name)
        dataset = set_time_metadata(
            dataset,
            name,
            time_bounds(dataset, name) if method != "point" else None,
            method,
        )
    if dataset.sizes["time"] == 1:
        dataset.time.attrs["stepsize"] = float(timestep)
    return dataset
