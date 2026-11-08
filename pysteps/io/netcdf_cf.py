"""
pysteps.io.netcdf_cf
====================

Read and write physical precipitation fields using the CF-1.8 conventions.

The bulk API supports floating fields and explicitly packed int16 fields.
Legacy forecast callbacks use the same schema and validated region writes.

.. autosummary::
    :toctree: ../generated/

    import_netcdf_cf
    export_netcdf_cf
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

from pysteps.dataset_metadata import (
    cf_grid_mapping,
    linked,
    normalize_dataset,
    physical_variables,
    set_time_metadata,
    temporal_method,
    unit_spec,
    utc_datetime,
)
from pysteps.exceptions import MissingOptionalDependency

__all__ = ["import_netcdf_cf", "export_netcdf_cf"]
_LINKS = ("bounds", "grid_mapping", "coordinates", "ancillary_variables")
_STORAGE = ("dtype", "_FillValue", "missing_value", "scale_factor", "add_offset")
_PACKING_FLOAT_DTYPE = np.dtype("float64")
_VALIDITY_ATTRIBUTES = (
    "_FillValue",
    "missing_value",
    "valid_range",
    "valid_min",
    "valid_max",
)


def _netcdf4():
    """Import netCDF4 or raise MissingOptionalDependency."""
    try:
        import netCDF4
    except ImportError as error:
        raise MissingOptionalDependency(
            "netCDF4 is required for CF NetCDF I/O; install the netCDF4 package "
            "in the Python environment running pySTEPS."
        ) from error
    return netCDF4


def _float_dtype(dtype):
    """Validate and return a float32 or float64 NumPy dtype."""
    requested = dtype
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"dtype must be float32/single or float64/double; received {requested!r}"
        ) from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(
            f"dtype must be float32/single or float64/double; received {dtype}"
        )
    return dtype


def _real_scalar(value):
    """Return whether a value is a real numeric scalar, before NumPy arithmetic."""
    return np.ndim(value) == 0 and np.asarray(value).dtype.kind in "iuf"


def _describe(values):
    """Describe dtype, shape and a bounded sample without dumping large arrays."""
    values = np.asarray(values)
    sample = values.ravel()[:6].tolist()
    return f"dtype={values.dtype}, shape={values.shape}, sample={sample!r}"


def _history(dataset, message):
    """Append a timestamped message to dataset history in place."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset.attrs["history"] = "\n".join(
        filter(None, [dataset.attrs.get("history"), f"{stamp}: {message}"])
    )


def _supply(attrs, updates, label):
    """Add missing attributes in place and reject conflicting definitions."""
    if not isinstance(updates, dict):
        raise ValueError(
            f"Metadata attributes for {label!r} must be a mapping; received "
            f"{type(updates).__name__}. Supply attribute names and values."
        )
    for key, value in updates.items():
        if key in attrs and attrs[key] is not None:
            if key == "projection":
                equal = pyproj.CRS(attrs[key]).equals(
                    pyproj.CRS(value), ignore_axis_order=True
                )
            else:
                equal = np.array_equal(attrs[key], value)
            if not equal:
                raise ValueError(
                    f"Metadata override conflicts with {label}.{key}: source "
                    f"{_describe(attrs[key])}; supplied {_describe(value)}. "
                    "metadata supplies missing definitions only; correct the source "
                    "or remove the conflicting override."
                )
        else:
            attrs[key] = value


def _metadata(dataset, metadata, variable, legacy):
    """Supply missing metadata explicitly; never overwrite a source definition."""
    if metadata is None:
        return dataset
    if not isinstance(metadata, dict) or set(metadata) - {
        "attrs",
        "variables",
        "coords",
    }:
        raise ValueError(
            "metadata accepts attrs, variables and coords mappings; received "
            f"{list(metadata) if isinstance(metadata, dict) else type(metadata).__name__}. "
            "Use only these keys, each with a mapping value."
        )
    for key, value in metadata.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"metadata[{key!r}] must be a mapping; received {type(value).__name__}"
            )
    dataset = dataset.copy(deep=False)
    _supply(dataset.attrs, metadata.get("attrs", {}), "dataset")
    for name, attrs in metadata.get("variables", {}).items():
        if name not in dataset:
            raise ValueError(
                f"Metadata refers to absent variable {name!r}; available variables: "
                f"{list(dataset.variables)}. Use an existing variable name in "
                "metadata['variables'], or correct the source file."
            )
        _supply(dataset[name].attrs, attrs, name)
    for name, supplied in metadata.get("coords", {}).items():
        if (
            name == "time"
            and not isinstance(supplied, (tuple, xr.DataArray))
            and np.ndim(supplied) == 0
        ):
            supplied = xr.DataArray(
                [utc_datetime(supplied)], dims="time", attrs={"standard_name": "time"}
            )
        candidate = xr.Dataset(coords={name: supplied})[name]
        if name in dataset:
            if candidate.dims != dataset[name].dims or not np.array_equal(
                candidate.values, dataset[name].values
            ):
                raise ValueError(
                    f"Metadata override conflicts with coordinate {name!r}: "
                    f"source dims={dataset[name].dims}, {_describe(dataset[name].values)}; "
                    f"supplied dims={candidate.dims}, {_describe(candidate.values)}. "
                    "Correct the source coordinate or remove the override."
                )
            _supply(dataset[name].attrs, candidate.attrs, name)
        else:
            dataset = dataset.assign_coords({name: candidate})
    candidates = physical_variables(dataset, legacy=legacy)
    if "time" in dataset.coords:
        for name in candidates:
            if set(dataset[name].dims) == {"x", "y"}:
                dataset[name] = dataset[name].expand_dims(time=dataset.time)
    # A supplied CRS is an explicit scientific definition, not a default datum.
    projection = metadata.get("attrs", {}).get("projection")
    if projection:
        supplied = cf_grid_mapping(projection)
        for name in candidates:
            mapping = linked(dataset[name], "grid_mapping") or "crs"
            if mapping not in dataset:
                dataset = dataset.assign_coords(
                    {mapping: xr.DataArray(np.int32(0), attrs=supplied.copy())}
                )
            else:
                for key, value in supplied.items():
                    dataset[mapping].attrs.setdefault(key, value)
            dataset[name].attrs["grid_mapping"] = mapping
    _history(dataset, "Supplied missing metadata: " + ", ".join(sorted(metadata)))
    return dataset


def _cf_coordinates(dataset, *, legacy=False):
    # Check links before coordinate decoding can discard a dangling reference.
    """Promote linked coordinates and expand scalar time and member axes."""
    coords = set()
    for name, var in dataset.variables.items():
        for key in _LINKS:
            value = linked(var, key)
            if value:
                references = value.split()
                if any(item not in dataset for item in references):
                    raise ValueError(
                        f"Dangling or unsupported {key} reference on {name!r}: "
                        f"{value!r}; unresolved tokens "
                        f"{[item for item in references if item not in dataset]}. "
                        "Supply the referenced variables or correct the link; "
                        "remove it only if it does not describe the data."
                    )
                if key != "ancillary_variables":
                    coords.update(references)
        if var.attrs.get("standard_name") in (
            "latitude",
            "longitude",
            "time",
            "realization",
            "forecast_reference_time",
        ):
            coords.add(name)
    dataset = dataset.set_coords(coords)
    candidates = physical_variables(dataset, legacy=legacy)
    scalar_times = [
        name
        for name in dataset.coords
        if dataset[name].ndim == 0
        and dataset[name].attrs.get("standard_name") == "time"
    ]
    if len(scalar_times) > 1 and any(
        not any(
            dim in dataset.coords and dataset[dim].attrs.get("standard_name") == "time"
            for dim in dataset[name].dims
        )
        for name in candidates
    ):
        raise ValueError(
            "Multiple scalar valid times require an explicit variable-specific time axis: "
            f"scalar coordinates={scalar_times}, physical fields={candidates}. "
            "Give each affected field a time dimension with standard_name='time' "
            "so its valid times are unambiguous."
        )
    for axis in list(dataset.coords):
        standard = dataset[axis].attrs.get("standard_name")
        if dataset[axis].ndim != 0 or standard not in ("time", "realization"):
            continue
        coordinate = dataset[axis]
        dataset = dataset.assign_coords(
            {
                axis: (
                    axis,
                    [
                        (
                            coordinate.values.item()
                            if standard == "realization"
                            else coordinate.values
                        )
                    ],
                    coordinate.attrs,
                )
            }
        )
        for name in candidates:
            if not any(
                dim in dataset.coords
                and dataset[dim].attrs.get("standard_name") == standard
                for dim in dataset[name].dims
            ):
                dataset[name] = dataset[name].expand_dims({axis: dataset[axis]})
    return dataset


def _same_numeric_type(left, right):
    """Return whether two NumPy dtypes represent the same NetCDF numeric type."""
    left, right = np.dtype(left), np.dtype(right)
    return left.kind == right.kind and left.itemsize == right.itemsize


def _validate_source_storage(dataset, source_encoding):
    """Validate CF packed-attribute types before using decoded source values."""
    for name, original in source_encoding.items():
        packed_dtype = np.dtype(original.get("dtype", dataset[name].dtype))
        terms = {
            key: original[key]
            for key in ("scale_factor", "add_offset")
            if key in original
        }
        term_dtypes = []
        for key, value in terms.items():
            if not _real_scalar(value) or not np.isfinite(value):
                raise ValueError(
                    f"Variable {name!r}: source scale_factor and add_offset must be "
                    f"finite scalars; {key} has {_describe(value)}. "
                    "Correct the source packing attribute before importing."
                )
            term_dtypes.append(np.asarray(value).dtype)
        if len(term_dtypes) == 2 and not _same_numeric_type(*term_dtypes):
            raise ValueError(
                f"Variable {name!r}: source scale_factor and add_offset must have "
                f"the same numeric type; received {dict(zip(terms, term_dtypes))}. "
                "Store both attributes with one type, such as float64."
            )
        if term_dtypes and not _same_numeric_type(term_dtypes[0], packed_dtype):
            if term_dtypes[0] not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError(
                    f"Variable {name!r}: source packing attributes must use the "
                    f"packed dtype ({packed_dtype}), float32, or float64; "
                    f"received {term_dtypes[0]}. Correct the source attribute types."
                )
            if packed_dtype.kind != "i" or packed_dtype.itemsize not in (1, 2, 4):
                raise ValueError(
                    f"Variable {name!r}: floating source packing attributes "
                    "require a byte, int16, or int32 variable in this CF-1.8 profile; "
                    f"received variable dtype={packed_dtype}, "
                    f"attribute dtype={term_dtypes[0]}. Re-encode the source "
                    "using a supported packed type or unpacked floating values."
                )

        attributes = dataset[name].attrs
        for key in _VALIDITY_ATTRIBUTES:
            if key in original:
                value = original[key]
            elif key in attributes:
                value = attributes[key]
            else:
                continue
            if not _same_numeric_type(np.asarray(value).dtype, packed_dtype):
                raise ValueError(
                    f"Variable {name!r}: source {key} must have the same numeric "
                    f"type as the packed variable ({packed_dtype}); "
                    f"received {_describe(value)}. Store the attribute in the "
                    "packed code domain with the variable's dtype."
                )


def _source_validity(dataset, source_encoding):
    """Apply declared raw valid ranges in the decoded affine domain exactly once."""
    dataset = dataset.copy(deep=False)
    for name, var in dataset.variables.items():
        keys = [
            key for key in ("valid_range", "valid_min", "valid_max") if key in var.attrs
        ]
        if not keys:
            continue
        if var.dtype.kind not in "iuf":
            raise ValueError(
                f"Variable {name!r}: valid ranges on nonnumeric decoded variables "
                f"are unsupported ({_describe(var.values)}). Use numeric data, "
                "or remove validity attributes that do not describe numeric limits."
            )
        attrs = var.attrs.copy()
        if "valid_range" in keys and len(keys) != 1:
            raise ValueError(
                f"Variable {name!r}: valid_range cannot be combined with "
                f"valid_min/valid_max; found {keys}. Keep valid_range alone, "
                "or use valid_min/valid_max instead."
            )
        original = source_encoding.setdefault(name, {})
        scale = original.get("scale_factor", 1.0)
        offset = original.get("add_offset", 0.0)
        if not np.isfinite([scale, offset]).all() or scale == 0:
            raise ValueError(
                f"Variable {name!r}: declared valid ranges need finite nonzero "
                f"source scaling; scale_factor={scale!r}, add_offset={offset!r}. "
                "Correct these source packing attributes."
            )
        lower, upper = -np.inf, np.inf
        if "valid_range" in keys:
            bounds = np.asarray(attrs["valid_range"])
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or bounds[0] > bounds[1]
            ):
                raise ValueError(
                    f"Variable {name!r}: valid_range must contain two finite "
                    f"ordered bounds; received {_describe(bounds)}. "
                    "Supply [minimum, maximum] in the stored code domain."
                )
            lower, upper = bounds
        else:
            for key in keys:
                if not np.isscalar(attrs[key]) or not np.isfinite(attrs[key]):
                    raise ValueError(
                        f"Variable {name!r}: valid_min/valid_max must be finite "
                        f"scalars; {key} has {_describe(attrs[key])}. "
                        "Supply one numeric bound in the stored code domain."
                    )
            lower, upper = attrs.get("valid_min", lower), attrs.get("valid_max", upper)
            if lower > upper:
                raise ValueError(
                    f"Variable {name!r}: valid_min is greater than valid_max "
                    f"({lower!r} > {upper!r}); correct the source bounds."
                )
        lo, hi = sorted([lower * scale + offset, upper * scale + offset])
        values = dataset[name].where((dataset[name] >= lo) & (dataset[name] <= hi))
        for key in keys:
            original[key] = attrs.pop(key)
            attrs["pysteps_source_" + key] = original[key]
        dataset[name] = (values.dims, values.data, attrs)
    return dataset


def import_netcdf_cf(
    filename,
    *,
    variable=None,
    group=None,
    dtype="double",
    load=True,
    legacy=False,
    metadata=None,
):
    """
    Read projected CF physical fields into an xarray Dataset.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the NetCDF file.
    variable : str, optional
        Active physical variable. Default to the dataset selector or sole
        physical field.
    group : str, optional
        NetCDF group to read. Default is the root group.
    dtype : str or numpy.dtype, optional
        Physical output precision: float32/'single' or float64/'double'. Default
        is 'double'.
    load : bool, optional
        Load arrays and close the file before returning. Default is True. False
        requires dask and leaves file ownership with the caller.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    metadata : dict, optional
        Missing definitions supplied through attrs, variables, and coords
        mappings. Existing definitions must agree with supplied values.

    Returns
    -------
    out : xarray.Dataset
        Decoded dataset in pySTEPS normalized physical units, with source storage
        recorded as provenance rather than output encoding.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    MissingOptionalDependency
        If netCDF4, or dask for lazy loading, is unavailable.

    Notes
    -----
    With load=False, close the returned dataset after use. Metadata are validated
    immediately; physical-value scans are deferred. Missing thresholds, times,
    interval durations, and CRS definitions are not inferred.
    """
    _netcdf4()
    dtype = _float_dtype(dtype)
    if not load:
        try:
            import dask.array  # noqa: F401
        except ImportError as error:
            raise MissingOptionalDependency(
                "Lazy CF import requires dask[array]; use load=True otherwise"
            ) from error
    source = xr.open_dataset(
        filename,
        group=group,
        engine="netcdf4",
        decode_coords=False,
        decode_times=xr.coders.CFDatetimeCoder(time_unit="us"),
        mask_and_scale=True,
        chunks=None if load else {},
    )
    try:
        if source.attrs.get("pysteps_write_status", "complete") != "complete":
            raise ValueError(
                f"NetCDF output {str(filename)!r} is incomplete: "
                f"pysteps_write_status={source.attrs['pysteps_write_status']!r}. "
                "Wait for the producer to finish or regenerate the forecast; "
                "do not relabel a partial file as complete."
            )
        source_encoding = {
            name: {key: var.encoding[key] for key in _STORAGE if key in var.encoding}
            for name, var in source.variables.items()
        }
        _validate_source_storage(source, source_encoding)
        dataset = _metadata(source, metadata, variable, legacy)
        dataset = _cf_coordinates(dataset, legacy=legacy)
        dataset = _source_validity(dataset, source_encoding)
        if load:
            dataset.load()
        dataset = normalize_dataset(
            dataset, variable, legacy=legacy, validate_values=load
        )
        for name in physical_variables(dataset):
            attrs = dataset[name].attrs.copy()
            values = dataset[name].astype(dtype)
            if load and np.isinf(values.values).any():
                raise ValueError(
                    f"Requested dtype {dtype} overflows physical values in {name!r}: "
                    f"source dtype={dataset[name].dtype}, physical bounds="
                    f"({np.nanmin(dataset[name].values)}, {np.nanmax(dataset[name].values)}); "
                    "use dtype='double' or check the source units and values."
                )
            dataset[name] = (values.dims, values.data, attrs)
            original = source_encoding.get(name, {})
            if "dtype" in original:
                dataset[name].attrs["pysteps_source_dtype"] = str(original["dtype"])
            for key in ("scale_factor", "add_offset", "_FillValue", "missing_value"):
                if key in original:
                    dataset[name].attrs["pysteps_source_" + key.lstrip("_")] = original[
                        key
                    ]
            # Retain source storage as provenance; export chooses its own encoding.
            dataset[name].encoding = {}
        dataset.encoding["pysteps_source_encoding"] = source_encoding
        if load:
            dataset.set_close(None)
        else:
            dataset.set_close(source.close)
        return dataset
    except Exception:
        source.close()
        raise
    finally:
        if load:
            source.close()


def _restore_validity(dataset):
    """Restore missing physical values from linked compatibility masks."""
    dataset = dataset.copy(deep=False)
    for name in physical_variables(dataset, legacy=True):
        attrs = dataset[name].attrs.copy()
        mask_name = attrs.pop("pysteps_validity_mask", None)
        if mask_name is None:
            continue
        if mask_name not in dataset:
            raise ValueError(
                f"Missing validity mask {mask_name!r} for {name!r}; restore the "
                "linked mask so missing pixels remain distinct from dry values."
            )
        if dataset[mask_name].dims != dataset[name].dims:
            raise ValueError(
                f"Misaligned validity mask {mask_name!r} for {name!r}: "
                f"mask dims={dataset[mask_name].dims}, field dims={dataset[name].dims}. "
                "Align the mask dimensions with the physical field before export."
            )
        mask = dataset[mask_name]
        if not np.isin(mask.values, [0, 1]).all():
            raise ValueError(
                f"Validity mask codes must be zero or one: {mask_name!r} for "
                f"{name!r} has {_describe(mask.values)}. Use 0 for missing "
                "and 1 for observed pixels."
            )
        attrs.pop("pysteps_fillna_value", None)
        values = dataset[name].where(mask == 1)
        dataset[name] = (values.dims, values.data, attrs)
    return dataset


def _subset(dataset, variables):
    """Select physical fields together with their transitive CF dependencies."""
    if variables is None:
        return dataset
    names = [variables] if isinstance(variables, str) else list(variables)
    candidates = physical_variables(dataset)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in candidates for name in names)
    ):
        raise ValueError(
            "variables must select distinct supported physical fields; "
            f"requested={names}, supported={candidates}. "
            "Choose a nonempty subset without duplicates."
        )
    keep = set(names)
    while True:
        before = keep.copy()
        for name in list(keep):
            if name not in dataset:
                raise ValueError(
                    f"Dangling CF dependency {name!r} while selecting {names}; "
                    "restore the referenced variable or correct its CF link."
                )
            var = dataset[name]
            keep.update(var.dims)
            for key in _LINKS:
                keep.update((linked(var, key) or "").split())
        dims = {dim for name in names for dim in dataset[name].dims}
        keep.update(
            name
            for name in dataset.coords
            if set(dataset[name].dims).issubset(dims) and name not in candidates
        )
        # A dimension such as nv need not have its own coordinate variable.
        keep = {name for name in keep if name in dataset or name not in dataset.dims}
        if keep == before:
            break
    selected = dataset.drop_vars(
        [name for name in dataset.variables if name not in keep]
    )
    selected.attrs = dataset.attrs.copy()
    if selected.attrs.get("precip_var") not in names:
        selected.attrs["precip_var"] = names[0]
    return selected


def _prepare(dataset, variables=None, include_latlon=False, validate_values=True):
    """Normalize and validate the export schema and physical metadata."""
    dataset = _subset(_restore_validity(dataset), variables)
    dataset = _cf_coordinates(dataset)
    dataset = normalize_dataset(dataset, validate_values=validate_values)
    for name in dataset.data_vars:
        dims = dataset[name].dims
        times = [
            dim
            for dim in dims
            if dim in dataset.coords
            and dataset[dim].attrs.get("standard_name") == "time"
        ]
        order = (
            [dim for dim in dims if dim not in (*times, "y", "x")]
            + times
            + [dim for dim in ("y", "x") if dim in dims]
        )
        dataset[name] = dataset[name].transpose(*order)
    dataset.attrs["Conventions"] = "CF-1.8"
    for key in ("title", "institution", "source", "references", "comment"):
        if key in dataset.attrs:
            if not isinstance(dataset.attrs[key], str):
                raise ValueError(
                    f"CF global attribute {key!r} must be a string; received "
                    f"{type(dataset.attrs[key]).__name__}. Supply descriptive text."
                )
            if not dataset.attrs[key].strip():
                dataset.attrs.pop(key)
    dataset.attrs.setdefault("title", "pySTEPS precipitation dataset")
    _history(
        dataset, "Exported physical precipitation with the pySTEPS CF NetCDF schema"
    )
    geographic = [
        name
        for name in dataset.coords
        if dataset[name].attrs.get("standard_name") in ("latitude", "longitude")
    ]
    standards = {dataset[name].attrs["standard_name"] for name in geographic}
    if include_latlon and standards != {"latitude", "longitude"}:
        projection = dataset.attrs["projection"]
        try:
            crs = pyproj.CRS(projection)
            source = crs.source_crs if crs.is_bound else crs
            xx, yy = np.meshgrid(dataset.x.values, dataset.y.values)
            lon, lat = pyproj.Transformer.from_crs(
                crs, source.geodetic_crs, always_xy=True
            ).transform(xx, yy, errcheck=True)
        except pyproj.exceptions.ProjError as error:
            raise ValueError(
                "Cannot derive longitude/latitude from the projected grid: "
                f"{error}. Check CRS {projection!r}, x/y coordinate units and "
                "grid extent against the projection's valid domain."
            ) from error
        # Finite x/y alone do not establish that the inverse projection exists.
        # Retain this check even when PROJ reports no transformation error.
        if not np.isfinite(lon).all() or not np.isfinite(lat).all():
            raise ValueError(
                "Grid has non-finite geographic coordinates in its CRS; "
                f"x extent={dataset.x.values[[0, -1]].tolist()}, "
                f"y extent={dataset.y.values[[0, -1]].tolist()}. "
                "Check the CRS, coordinate units and projection domain."
            )
        additions = {}
        for name, standard, values, unit in (
            ("lon", "longitude", lon, "degrees_east"),
            ("lat", "latitude", lat, "degrees_north"),
        ):
            if standard not in standards:
                if name in dataset:
                    raise ValueError(
                        f"Geographic coordinate name collision: {name!r} already "
                        f"exists with standard_name={dataset[name].attrs.get('standard_name')!r}. "
                        "Rename that variable or supply a geographic coordinate "
                        f"with standard_name={standard!r}."
                    )
                additions[name] = (
                    ("y", "x"),
                    values,
                    {"standard_name": standard, "units": unit},
                )
        dataset = dataset.assign_coords(additions)
    elif not include_latlon:
        dataset = dataset.drop_vars(geographic)
    for name in dataset.variables:
        var = dataset[name]
        for key in _LINKS:
            value = linked(var, key)
            if value:
                if key == "coordinates":
                    value = " ".join(
                        token for token in value.split() if token in dataset
                    )
                if value:
                    var.attrs[key] = value
                else:
                    var.attrs.pop(key, None)
        # Values have already been decoded and normalized. Source packing may
        # describe different units or precision, and retaining it would apply
        # that old storage contract to new physical values. Output encoding is
        # chosen explicitly by _encoding(); source storage remains provenance.
        # Time units/calendar are serialization choices, so retain those here.
        var.encoding = (
            {
                key: var.encoding[key]
                for key in ("units", "calendar", "dtype", "_FillValue")
                if key in var.encoding
            }
            if np.issubdtype(var.dtype, np.datetime64)
            else {}
        )
        var.attrs.pop("_FillValue", None)
        var.attrs.pop("missing_value", None)
        var.attrs.pop("scale_factor", None)
        var.attrs.pop("add_offset", None)
        var.attrs.pop("_Unsigned", None)
        # This optional statistic is derived from values and becomes stale
        # after unit/dtype changes or quantization. Do not mislabel the output.
        var.attrs.pop("actual_range", None)
        # pySTEPS normalizes rate to mm/h internally; spell the file unit using
        # the conventional CF form. These are not CF's canonical m s-1 units.
        if var.attrs.get("units") == "mm/h":
            var.attrs["units"] = "mm h-1"
        auxiliary = [
            coord
            for coord in dataset.coords
            if coord != name
            and coord not in var.dims
            and set(dataset[coord].dims).issubset(var.dims)
            and dataset[coord].attrs.get("grid_mapping_name") is None
        ]
        if name in dataset.data_vars and auxiliary:
            var.attrs["coordinates"] = " ".join(
                sorted(
                    set((var.attrs.get("coordinates") or "").split()) | set(auxiliary)
                )
            )
    dataset.encoding = {}
    return dataset


def _encoding(dataset, encoding=None, *, legacy=False):
    """Validate storage options and build explicit per-variable encodings."""
    encoding = encoding or {}
    if set(encoding) - set(dataset.variables):
        raise ValueError(
            "Encoding refers to variables outside the export: "
            f"{sorted(set(encoding) - set(dataset.variables))}. "
            f"Available variables: {list(dataset.variables)}. "
            "Remove encoding entries for excluded or nonexistent variables."
        )
    physical = physical_variables(dataset)
    result = {}
    allowed = {
        "dtype",
        "_FillValue",
        "zlib",
        "complevel",
        "shuffle",
        "chunksizes",
        "fletcher32",
        "scale_factor",
        "add_offset",
    }
    for name, var in dataset.variables.items():
        requested = dict(encoding.get(name, {}))
        if set(requested) - allowed:
            raise ValueError(
                f"Unsupported encoding options for {name!r}: "
                f"{sorted(set(requested) - allowed)}; supported options: {sorted(allowed)}"
            )
        if (
            not legacy
            and name not in physical
            and any(key in requested for key in ("scale_factor", "add_offset"))
        ):
            raise ValueError(
                "scale_factor/add_offset are supported only for physical fields; "
                f"remove them from encoding[{name!r}]. Physical fields: {physical}"
            )
        datetime = np.issubdtype(var.dtype, np.datetime64)
        flags = "flag_values" in var.attrs or "flag_masks" in var.attrs
        default_dtype = var.dtype
        if flags:
            default_dtype = np.asarray(
                var.attrs.get("flag_values", var.attrs.get("flag_masks"))
            ).dtype
        requested_dtype = requested.get(
            "dtype", "float64" if datetime else default_dtype
        )
        try:
            dtype = np.dtype(requested_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid encoding dtype for {name!r}: {requested_dtype!r}; "
                "supply a NumPy numeric dtype such as float32, float64 or int16."
            ) from error
        if name in physical:
            if not legacy:
                if dtype.kind == "i":
                    if dtype != np.dtype("int16"):
                        raise ValueError(
                            f"Integer physical storage requires int16 for {name!r}; "
                            f"received {dtype}. Use int16 with an explicit "
                            "scale_factor, or float32/float64 without packing."
                        )
                    if "scale_factor" not in requested:
                        raise ValueError(
                            f"Integer physical storage for {name!r} requires an "
                            "explicit scale_factor in pySTEPS normalized units "
                            f"({var.attrs.get('units')}); for example, 0.1 for "
                            "0.1-unit steps."
                        )
                elif dtype.kind != "f" or any(
                    key in requested for key in ("scale_factor", "add_offset")
                ):
                    raise ValueError(
                        f"Packing options require int16 physical storage for {name!r}; "
                        f"received dtype={dtype}, packing keys="
                        f"{[key for key in ('scale_factor', 'add_offset') if key in requested]}. "
                        "Use int16 with scale_factor, or float32/float64 without "
                        "scale_factor/add_offset."
                    )
            if dtype.kind == "f" and dtype.itemsize not in (4, 8):
                raise ValueError(
                    f"Physical floating storage must be float32 or float64; "
                    f"{name!r} requested {dtype}. Choose either supported precision."
                )
            options = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "chunksizes": tuple(
                    1 if dim not in ("x", "y") else min(256, var.sizes[dim])
                    for dim in var.dims
                ),
                "_FillValue": (
                    np.nan
                    if dtype.kind == "f"
                    else (
                        -32768
                        if dtype == np.dtype("int16") and not legacy
                        else _netcdf4().default_fillvals[dtype.str[1:]]
                    )
                ),
            }
        else:
            if dtype.kind in "bu" or dtype.kind == "i" and dtype.itemsize > 4:
                values = var.values
                limit = np.iinfo("int32")
                if np.any(values < limit.min) or np.any(values > limit.max):
                    raise ValueError(
                        f"{name!r} is outside the CF-1.8 int32 profile: "
                        f"{_describe(values)}. Use values between {limit.min} "
                        f"and {limit.max} for integer coordinates/flags."
                    )
                dtype = np.dtype("int32")
            fill = (
                _netcdf4().default_fillvals[dtype.str[1:]]
                if flags and dtype.kind == "i"
                else np.nan
                if name in dataset.data_vars and dtype.kind == "f"
                else None
            )
            options = {"_FillValue": fill}
        if var.attrs.get("standard_name") == "realization":
            labels = var.values
            limits = np.iinfo("int32")
            if (
                not np.isfinite(labels).all()
                or np.any(labels != np.floor(labels))
                or len(np.unique(labels)) != labels.size
                or np.any(labels < limits.min)
                or np.any(labels > limits.max)
            ):
                raise ValueError(
                    "Realization labels must be unique integers in int32 range; "
                    f"{name!r} has {_describe(labels)}. Use distinct integer member "
                    f"labels between {limits.min} and {limits.max}."
                )
            if "dtype" in requested and dtype != np.dtype("int32"):
                raise ValueError(
                    f"Realization coordinates require int32 storage; {name!r} "
                    f"requested {dtype}. Use dtype='int32' or omit the override."
                )
            dtype = np.dtype("int32")
        if name in ("x", "y") and dtype != np.dtype("float64"):
            raise ValueError(
                f"Projected coordinates require float64 storage; {name!r} "
                f"requested {dtype}. Use dtype='float64' or omit the override."
            )
        if (
            (dtype.kind == "i" and dtype.itemsize not in (1, 2, 4))
            or (dtype.kind == "f" and dtype.itemsize not in (4, 8))
            or dtype.kind not in "if"
        ):
            raise ValueError(
                f"Unsupported CF numeric dtype for {name!r}: {dtype}. "
                "Use int8/int16/int32 or float32/float64, subject to the field's "
                "coordinate and packing requirements."
            )
        options.update(requested)
        options["dtype"] = dtype
        if not legacy and name in physical and dtype == np.dtype("int16"):
            options.setdefault("add_offset", _PACKING_FLOAT_DTYPE.type(0.0))
            if options.get("_FillValue") is None:
                raise ValueError(
                    f"Int16 physical packing requires a reserved _FillValue for "
                    f"{name!r}; received None. Omit this option to use -32768, "
                    "or choose a noncolliding int16 code. This explicit missing "
                    "code is a pySTEPS writer policy, not a CF requirement."
                )
        if datetime and dtype != np.dtype("float64"):
            raise ValueError(
                f"CF times require float64 storage in this writer; {name!r} "
                f"requested {dtype}. Use dtype='float64' or omit the override."
            )
        if (
            not isinstance(options.get("complevel", 4), (int, np.integer))
            or not 0 <= options.get("complevel", 4) <= 9
        ):
            raise ValueError(
                f"complevel must be an integer from zero to nine for {name!r}; "
                f"received {options['complevel']!r}"
            )
        for key in ("zlib", "shuffle", "fletcher32"):
            if key in options and not isinstance(options[key], (bool, np.bool_)):
                raise ValueError(
                    f"{key} must be a boolean for {name!r}; received {options[key]!r}. "
                    "Use True or False."
                )
        if "chunksizes" in options:
            chunks = options["chunksizes"]
            if (
                not isinstance(chunks, (list, tuple, np.ndarray))
                or np.ndim(chunks) != 1
                or len(chunks) != var.ndim
                or any(
                    not isinstance(size, (int, np.integer))
                    or size <= 0
                    or size > length
                    for size, length in zip(chunks, var.shape)
                )
            ):
                raise ValueError(
                    f"Invalid chunksizes for {name!r}: {chunks!r}; "
                    f"dims={var.dims}, shape={var.shape}. Supply one positive "
                    "integer per dimension, no larger than its dimension length."
                )
        for key, default in (("scale_factor", 1.0), ("add_offset", 0.0)):
            value = options.get(key, default)
            if (
                not _real_scalar(value)
                or not np.isfinite(value)
                or key == "scale_factor"
                and value <= 0
            ):
                raise ValueError(
                    f"Invalid {key} for {name!r}: {_describe(value)}. "
                    "Supply a finite real scalar"
                    + (" greater than zero." if key == "scale_factor" else ".")
                )
        fill = options.get("_FillValue")
        if fill is not None:
            cast = None
            if _real_scalar(fill):
                with np.errstate(over="ignore", invalid="ignore"):
                    cast = np.asarray(fill).astype(dtype).item()
            if (
                cast is None
                or np.isinf(fill)
                or (np.isfinite(fill) and not np.isfinite(cast))
                or dtype.kind == "i"
                and cast != fill
            ):
                raise ValueError(
                    f"Unrepresentable fill value for {name!r}: {_describe(fill)}; "
                    f"output dtype={dtype}. Choose a scalar exactly representable "
                    "in that dtype, outside the valid stored-code range."
                )
            options["_FillValue"] = cast
        for key in ("scale_factor", "add_offset"):
            if key in options:
                # CF requires both attributes to have one unpacked numeric type.
                # Standardize exported packing on NetCDF double precision.
                options[key] = _PACKING_FLOAT_DTYPE.type(options[key])
        result[name] = options
    return result


def _valid_limits(attrs, *, name=None):
    """Return physical validity limits, or None when undeclared."""
    keys = [key for key in ("valid_range", "valid_min", "valid_max") if key in attrs]
    if not keys:
        return None
    if "valid_range" in keys:
        limits = np.asarray(attrs["valid_range"])
        if len(keys) != 1 or limits.shape != (2,):
            raise ValueError(
                f"Variable {name!r}: valid_range must be two bounds without "
                f"valid_min/valid_max; received {_describe(limits)}, keys={keys}. "
                "Use [minimum, maximum] alone or separate valid_min/valid_max."
            )
        low, high = limits
    else:
        low, high = attrs.get("valid_min", -np.inf), attrs.get("valid_max", np.inf)
    if (
        not _real_scalar(low)
        or not _real_scalar(high)
        or np.isnan(low)
        or np.isnan(high)
        or low > high
    ):
        raise ValueError(
            f"Invalid physical valid range for {name!r}: low={low!r}, high={high!r}. "
            "Use ordered numeric bounds without NaN, in pySTEPS normalized units."
        )
    return float(low), float(high)


def _storage_attributes(attrs, options, *, name=None):
    """Convert physical validity attributes to the stored code domain."""
    attrs = attrs.copy()
    limits = _valid_limits(attrs, name=name)
    if limits is None:
        return attrs
    scale = options.get("scale_factor", 1.0)
    offset = options.get("add_offset", 0.0)
    dtype = options["dtype"]
    code_limits = np.iinfo(dtype) if dtype.kind == "i" else np.finfo(dtype)
    converted = []
    with np.errstate(over="ignore", invalid="ignore"):
        for value in limits:
            code = (value - offset) / scale
            if dtype.kind == "i":
                code = np.rint(code)
            converted.append(np.clip(code, code_limits.min, code_limits.max))
    low, high = converted
    fill = options.get("_FillValue")
    if dtype.kind == "i" and fill is not None:
        low += low == fill
        high -= high == fill
    if low > high:
        raise ValueError(
            f"Valid range has no representable nonmissing code for {name!r}: "
            f"physical bounds={limits}, dtype={dtype}, scale_factor={scale}, "
            f"add_offset={offset}, _FillValue={fill}. "
            "Choose packing that covers the declared physical range."
        )
    if fill is not None and np.isfinite(fill) and low <= fill <= high:
        raise ValueError(
            f"The fill code must be outside the stored valid range for {name!r}: "
            f"_FillValue={fill}, stored bounds=({low}, {high}). "
            "Choose a fill code outside those bounds or adjust the packing."
        )
    for key in ("valid_range", "valid_min", "valid_max"):
        if key not in attrs:
            continue
        if key == "valid_range":
            attrs[key] = np.asarray([low, high], dtype=dtype)
        else:
            attrs[key] = np.asarray(low if key == "valid_min" else high, dtype=dtype)[
                ()
            ]
    return attrs


def _stored_values(values, options, *, quantity=None, valid_limits=None, name=None):
    """Encode physical values while preserving masks and checking code ranges."""
    if np.ma.isMaskedArray(values):
        if values.dtype.kind not in "iuf":
            raise ValueError(
                f"Physical data must be finite numbers or NaN for {name!r}; "
                f"received dtype={values.dtype}. Convert to a real numeric array."
            )
        # Preserve integer-array masks before conversion: NaN cannot represent
        # missing values in the original integer dtype.
        values = values.astype("float64").filled(np.nan)
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise ValueError(
            f"Physical data must be finite numbers or NaN for {name!r}; "
            f"received dtype={values.dtype}. Convert to a real numeric array."
        )
    if np.isinf(values).any():
        raise ValueError(
            f"Physical data must be finite numbers or NaN for {name!r}; "
            f"found {np.count_nonzero(np.isinf(values))} infinite values. "
            "Correct invalid values or mark genuinely missing pixels as NaN."
        )
    if quantity is not None and quantity != "reflectivity" and np.any(values < 0):
        raise ValueError(
            f"Negative physical rain rate/depth in {name!r}: minimum={np.nanmin(values)}; "
            "explicitly invert modelling transforms first, or correct the source values."
        )
    if valid_limits is not None and (
        np.any(values < valid_limits[0]) or np.any(values > valid_limits[1])
    ):
        raise ValueError(
            f"Physical values are outside their declared valid range for {name!r}: "
            f"observed=({np.nanmin(values)}, {np.nanmax(values)}), "
            f"declared={valid_limits}. Correct the values or their physical bounds."
        )
    scale = options.get("scale_factor", 1.0)
    offset = options.get("add_offset", 0.0)
    dtype = options["dtype"]
    fill = options.get("_FillValue")
    if (
        values.dtype == dtype
        and dtype.kind == "f"
        and scale == 1
        and offset == 0
        and fill is not None
        and np.isnan(fill)
    ):
        return values
    missing = np.isnan(values)
    with np.errstate(over="ignore", invalid="ignore"):
        stored = (values.astype("float64") - offset) / scale
        if dtype.kind == "i":
            stored = np.rint(stored)
            limit = np.iinfo(dtype)
        else:
            limit = np.finfo(dtype)
        valid = stored[~missing]
        if (
            not np.isfinite(valid).all()
            or np.any(valid < limit.min)
            or np.any(valid > limit.max)
        ):
            raise ValueError(
                f"Values overflow the requested NetCDF encoding for {name!r}: "
                f"stored-code range=({valid.min()}, {valid.max()}), dtype={dtype}, "
                f"scale_factor={scale}, add_offset={offset}. Choose a larger "
                "packing step, compatible offset, or floating storage."
            )
        fill = options.get("_FillValue")
        if fill is None:
            if missing.any():
                raise ValueError(
                    f"Missing physical values require an explicit output fill value "
                    f"for {name!r}; omit _FillValue to use the writer default, "
                    "or choose a noncolliding code."
                )
            fill = 0
        cast = stored[~missing].astype(dtype)
        if (
            options.get("_FillValue") is not None
            and np.isfinite(fill)
            and np.any(cast == fill)
        ):
            raise ValueError(
                f"Valid physical values collide with the output fill value for "
                f"{name!r}: _FillValue={fill}, dtype={dtype}, scale_factor={scale}, "
                f"add_offset={offset}. Choose a reserved code outside the data range."
            )
        if quantity is not None and cast.size:
            decoded_limits = (
                np.asarray([cast.min(), cast.max()], dtype="float64") * scale + offset
            )
            if not np.isfinite(decoded_limits).all():
                raise ValueError(
                    f"Decoded physical values overflow the requested encoding for "
                    f"{name!r}: scale_factor={scale}, add_offset={offset}, "
                    f"decoded bounds={decoded_limits.tolist()}. "
                    "Choose packing with finite decoded physical values."
                )
            if quantity != "reflectivity" and np.any(decoded_limits < 0):
                raise ValueError(
                    f"Encoding would produce negative physical rain for {name!r}: "
                    f"decoded bounds={decoded_limits.tolist()}, scale_factor={scale}, "
                    f"add_offset={offset}; choose a compatible offset."
                )
        stored = np.where(missing, fill, stored).astype(dtype)
    return stored


def _coordinate_data(var):
    """Encode coordinate values and attributes without physical-field packing."""
    attrs = var.attrs.copy()
    if np.issubdtype(var.dtype, np.datetime64):
        # NetCDF stores time as numeric offsets from an epoch. Work in integer
        # microseconds first so conversion to float64 seconds is the only place
        # where time precision can be lost.
        values = var.values.astype("datetime64[us]")
        if np.isnat(values).any():
            raise ValueError(
                f"Missing datetime coordinate {var.name!r}; supply actual valid "
                "times for every coordinate entry before export."
            )

        # Preserve an explicit seconds-based epoch when one is supplied;
        # otherwise anchor the encoding at the earliest coordinate. Using a
        # nearby epoch keeps float64 offsets small and maximizes precision.
        units = var.encoding.get(
            "units", "seconds since " + str(values.min()).replace("T", " ")
        )
        origin = np.datetime64(units.removeprefix("seconds since "), "us")
        offsets = (values - origin) / np.timedelta64(1, "us")
        numeric = offsets.astype("float64") / 1e6

        # Reconstruct the datetimes before writing. This prevents a file from
        # being published if float64 seconds cannot preserve microsecond input.
        restored = origin + np.rint(numeric * 1e6).astype("timedelta64[us]")
        if not np.array_equal(values, restored):
            raise ValueError(
                f"Float64 seconds cannot preserve required time precision for "
                f"{var.name!r} with units={units!r}. Choose a seconds-based epoch "
                "closer to the coordinate dates in the time encoding."
            )
        attrs.update(units=units, calendar="proleptic_gregorian")
        return numeric, attrs
    return var.values, attrs


class _CFWriter:
    """Write validated batches to a temporary file and publish on completion.

    The legacy forecast-export API can deliver one timestep or one ensemble
    member per callback instead of a complete xarray Dataset. The writer
    therefore creates the complete schema and fixed coordinates first, then
    fills one region of each physical variable per callback. The bulk API uses
    the same path so packing, validation, and failure behaviour do not diverge.
    xarray.to_netcdf supports Dask computation but does not expose successive
    callback-driven writes into regions of an existing NetCDF variable.

    Partial output exists only under a hidden temporary name. ``finish``
    publishes it after every expected batch has arrived; any failed or
    incomplete write calls ``abort`` and removes the temporary file.
    """

    def __init__(
        self, dataset, filename, encoding, *, overwrite=False, incremental=None
    ):
        """Validate auxiliary values and initialize the temporary NetCDF schema."""
        self.dataset, self.encoding = dataset, encoding
        self.filename = Path(filename)
        self.overwrite, self.incremental = overwrite, incremental
        self.physical = physical_variables(dataset)
        self.count, self.closed = 0, False
        self._code_ranges = {}
        # Bulk export has one complete batch. Incremental export receives one
        # batch for each position on the selected callback axis.
        self.expected = (
            1
            if incremental is None
            else dataset.sizes["time" if incremental == "timestep" else "ens_number"]
        )
        self.path, self.ncfile = None, None
        if self.filename.exists() and not overwrite:
            raise FileExistsError(
                f"Output {str(self.filename)!r} already exists; choose a new filename "
                "or set overwrite=True to replace it on successful completion."
            )
        # Encode small coordinates/auxiliaries before any file is created.
        prepared = {}
        for name, var in dataset.variables.items():
            if name not in self.physical:
                values, attrs = _coordinate_data(var)
                for key in ("flag_values", "flag_masks"):
                    if key in attrs:
                        original = np.asarray(attrs[key])
                        cast = original.astype(encoding[name]["dtype"])
                        if not np.isfinite(original).all() or not np.array_equal(
                            original, cast
                        ):
                            raise ValueError(
                                f"Flag codes are outside the output dtype range for "
                                f"{name!r}.{key}: {_describe(original)}, "
                                f"output dtype={encoding[name]['dtype']}. "
                                "Use exactly representable integer flag codes."
                            )
                        if original.ndim != 1 or len(original) != len(
                            attrs.get("flag_meanings", "").split()
                        ):
                            raise ValueError(
                                f"Flag codes and meanings must have matching lengths "
                                f"for {name!r}: {key} shape={original.shape}, "
                                f"flag_meanings={attrs.get('flag_meanings')!r}. "
                                "Supply a one-dimensional code array and one "
                                "space-separated meaning per code."
                            )
                        attrs[key] = cast
                # Mask auxiliary missing values before casting flag codes.
                prepared[name] = (
                    _stored_values(values, encoding[name], name=name),
                    attrs,
                )
        for parent in dataset.variables:
            bound = linked(dataset[parent], "bounds")
            if bound in prepared:
                bound_attrs = prepared[bound][1]
                parent_attrs = prepared[parent][1]
                # CF bounds inherit units and calendar from their coordinate.
                # Check that independently encoded values agree, then avoid
                # duplicating those attributes on the bounds variable.
                for key in ("units", "calendar"):
                    if key in bound_attrs and key in parent_attrs:
                        if bound_attrs[key] != parent_attrs[key]:
                            raise ValueError(
                                f"Bounds and coordinate serialization disagree: "
                                f"{bound!r}.{key}={bound_attrs[key]!r}, "
                                f"{parent!r}.{key}={parent_attrs[key]!r}. "
                                "Use matching time units and calendar encodings "
                                "for the coordinate and its bounds."
                            )
                        bound_attrs.pop(key)
        try:
            # Region writes must never expose a half-written forecast at the
            # requested destination. Create the complete schema under a hidden
            # sibling name and publish it atomically only in finish().
            fd, name = tempfile.mkstemp(
                prefix="." + self.filename.name + ".incomplete-",
                suffix=".nc",
                dir=self.filename.parent,
            )
            os.close(fd)
            self.path = Path(name)
            self.ncfile = _netcdf4().Dataset(self.path, "w", format="NETCDF4")
            self.ncfile.setncatts(dataset.attrs)
            self.ncfile.pysteps_write_status = "incomplete"
            for dim, size in dataset.sizes.items():
                self.ncfile.createDimension(dim, size)
            for name, var in dataset.variables.items():
                options = encoding[name].copy()
                dtype, fill = options.pop("dtype"), options.pop("_FillValue", None)
                scale = options.pop("scale_factor", None)
                offset = options.pop("add_offset", None)
                output = self.ncfile.createVariable(
                    name,
                    dtype,
                    var.dims,
                    fill_value=fill if fill is not None else False,
                    **options,
                )
                attrs = (
                    _storage_attributes(var.attrs, encoding[name], name=name)
                    if name in self.physical
                    else prepared[name][1]
                )
                if scale is not None:
                    attrs["scale_factor"] = scale
                if offset is not None:
                    attrs["add_offset"] = offset
                output.setncatts(attrs)
                output.set_auto_maskandscale(False)
                if name not in self.physical:
                    output[...] = np.asarray(prepared[name][0], dtype=dtype)
        except Exception:
            self.abort()
            raise

    def write(self, arrays):
        """Validate and write the next complete batch of physical values."""
        try:
            if self.closed:
                raise ValueError(
                    f"Writer is closed for {str(self.filename)!r}; initialize "
                    "a new exporter before writing another forecast."
                )
            if self.count >= self.expected:
                raise ValueError(
                    f"All {self.expected} forecast batches have been written to "
                    f"{str(self.filename)!r}; close the exporter instead of "
                    "sending another batch."
                )
            if self.incremental is not None and len(self.physical) != 1:
                raise ValueError(
                    "Legacy incremental output requires one forecast field; "
                    f"selected={self.physical}. Select one field for callbacks "
                    "or use bulk export for multiple fields."
                )
            if not isinstance(arrays, dict):
                arrays = {self.physical[0]: arrays}
            if set(arrays) != set(self.physical):
                raise ValueError(
                    "A batch must contain every selected physical variable; "
                    f"expected={self.physical}, received={list(arrays)}. "
                    "Supply exactly one array per selected field."
                )
            prepared = {}
            for name, values in arrays.items():
                var = self.dataset[name]
                # In incremental mode this callback owns one complete slice on
                # either the time or realization axis. Other dimensions are
                # written in full. Bulk mode selects the entire variable.
                axis = "time" if self.incremental == "timestep" else "ens_number"
                index = tuple(
                    (
                        self.count
                        if dim == axis and self.incremental is not None
                        else slice(None)
                    )
                    for dim in var.dims
                )
                shape = tuple(
                    size
                    for dim, size in zip(var.dims, var.shape)
                    if self.incremental is None or dim != axis
                )
                if np.shape(values) != shape:
                    raise ValueError(
                        f"field has invalid shape for {name!r}: {np.shape(values)} "
                        f"!= {shape}; field dims={var.dims}, "
                        f"incremental={self.incremental!r}, batch={self.count}. "
                        "Supply the full field for bulk export, or one complete "
                        "slice on the selected callback axis."
                    )
                prepared[name] = (
                    index,
                    _stored_values(
                        values,
                        self.encoding[name],
                        quantity=unit_spec(var.attrs["units"])[0],
                        valid_limits=_valid_limits(var.attrs, name=name),
                        name=name,
                    ),
                )
            for name, (_, values) in prepared.items():
                options = self.encoding[name]
                fill = options.get("_FillValue")
                if (
                    fill is None
                    or not np.isfinite(fill)
                    or _valid_limits(self.dataset[name].attrs, name=name) is not None
                ):
                    continue
                # Endpoint integer fill codes imply a safe default range.
                if options["dtype"].kind == "i":
                    limit = np.iinfo(options["dtype"])
                    if fill in (limit.min, limit.max):
                        continue
                valid = values != fill
                if valid.any():
                    low, high = float(values[valid].min()), float(values[valid].max())
                    previous = self._code_ranges.get(name, (low, high))
                    low, high = min(low, previous[0]), max(high, previous[1])
                    if low <= fill <= high:
                        raise ValueError(
                            f"The fill code must be outside the stored valid range "
                            f"for {name!r}: _FillValue={fill}, "
                            f"stored bounds=({low}, {high}). Choose a fill code "
                            "outside the range across all forecast batches."
                        )
                    self._code_ranges[name] = (low, high)
            # Validate and encode every physical variable in the batch before
            # performing any region write. A failure therefore aborts the
            # unpublished file without leaving a partially accepted batch.
            for name, (index, values) in prepared.items():
                self.ncfile[name][index] = values
            self.count += 1
            self.ncfile.sync()
        except Exception:
            self.abort()
            raise

    def finish(self):
        """Publish the completed file atomically and return its Path."""
        if self.closed:
            raise ValueError(
                f"Writer is already closed for {str(self.filename)!r}; "
                "initialize a new exporter for another output."
            )
        try:
            if self.count != self.expected:
                raise ValueError(
                    f"Incomplete forecast: wrote {self.count} of {self.expected} "
                    f"batches for {str(self.filename)!r} (incremental={self.incremental!r}). "
                    "Regenerate the forecast with every expected callback batch "
                    "before closing; partial output is discarded."
                )
            for name, (low, high) in self._code_ranges.items():
                fill = self.encoding[name]["_FillValue"]
                # CF/NUG readers may infer a one-sided range from the fill.
                # Declare an explicit range when that would hide real values.
                if (fill <= 0 and low < fill) or (fill > 0 and high > fill):
                    self.ncfile[name].setncattr(
                        "valid_range",
                        np.asarray([low, high], dtype=self.encoding[name]["dtype"]),
                    )
            self.ncfile.pysteps_write_status = "complete"
            self.ncfile.close()
            self.ncfile = None
            if self.overwrite:
                os.replace(self.path, self.filename)
            else:
                # Publish without replacing a destination created by another
                # process after the initial existence check.
                os.link(self.path, self.filename)
                self.path.unlink()
            self.closed = True
            return self.filename
        except Exception:
            self.abort()
            raise

    def abort(self):
        """Close the temporary file and remove unpublished output."""
        if self.ncfile is not None:
            self.ncfile.close()
            self.ncfile = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self.closed = True


def export_netcdf_cf(
    dataset,
    filename,
    *,
    variables=None,
    encoding=None,
    include_latlon=False,
    overwrite=False,
):
    """
    Export physical fields atomically as CF-1.8 NETCDF4.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    filename : str or pathlib.Path
        Destination path. Its parent directory must exist.
    variables : str or sequence of str, optional
        Physical fields to export, retaining their CF dependencies. Default is
        all supported physical fields.
    encoding : dict, optional
        Per-variable storage options keyed by variable name. Accept dtype,
        _FillValue, scale_factor, add_offset, zlib, complevel, shuffle,
        chunksizes, and fletcher32.
    include_latlon : bool, optional
        Include geographic coordinates derived from the projected grid. Default
        is False.
    overwrite : bool, optional
        Replace an existing destination on successful completion. Default is
        False.

    Returns
    -------
    out : pathlib.Path
        Path of the completed file.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    FileExistsError
        If the destination exists and overwrite is False.
    MissingOptionalDependency
        If netCDF4 is unavailable.

    Notes
    -----
    Preserve floating precision by default, using zlib level 4 and shuffle. Physical
    int16 storage requires an explicit positive scale_factor in pySTEPS normalized
    units (mm/h, mm or dBZ).
    Both scale_factor and add_offset are stored as float64; add_offset defaults to
    zero and _FillValue to -32768. Packing uses stored =
    round((physical - add_offset) / scale_factor). Preserve missing masks before
    quantization and reject overflow or fill collisions. Invert modelling transforms
    before export. The bulk and incremental forecast APIs share a low-level writer
    because the latter can supply one timestep or ensemble member per callback.
    Partial output remains under a temporary name and is published only after all
    expected batches have been written successfully.
    """
    _netcdf4()
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise FileExistsError(
            f"Output {str(filename)!r} already exists; choose a new filename "
            "or set overwrite=True to replace it on successful completion."
        )
    prepared = _prepare(dataset, variables, include_latlon)
    options = _encoding(prepared, encoding)
    # Validate the schema before creation, then each batch before region writes.
    writer = _CFWriter(prepared, filename, options, overwrite=overwrite)
    try:
        writer.write({name: prepared[name].values for name in writer.physical})
        return writer.finish()
    except Exception:
        writer.abort()
        raise


def _legacy_forecast_writer(
    outpath,
    outfnprefix,
    startdate,
    timestep,
    n_timesteps,
    shape,
    metadata,
    n_ens_members=1,
    datatype=np.float32,
    incremental=None,
    fill_value=None,
    scale_factor=None,
    offset=None,
    complevel=9,
    **kwargs,
):
    """Adapt legacy forecast options to the shared CF schema and batch writer."""
    from pysteps.xarray_helpers import convert_input_to_xarray_dataset

    if incremental not in (None, "timestep", "member"):
        raise ValueError(
            "incremental must be None, 'timestep' or 'member'; "
            f"received {incremental!r}. Choose the callback axis or None for bulk export."
        )
    if unit_spec(metadata["unit"])[2] != 1:
        raise ValueError(
            "Legacy NetCDF exporter requires pySTEPS normalized physical values; "
            f"received unit={metadata['unit']!r}; normalize units before export "
            "by converting values and metadata to mm/h, mm or dBZ."
        )
    if not _real_scalar(timestep) or not np.isfinite(timestep) or timestep <= 0:
        raise ValueError(
            f"timestep must be positive minutes; received {timestep!r}. "
            "Supply the sampling cadence as a finite positive number."
        )
    leads = np.asarray(
        n_timesteps if isinstance(n_timesteps, list) else np.arange(1, n_timesteps + 1),
        dtype=float,
    )
    if (
        leads.ndim != 1
        or not len(leads)
        or not np.isfinite(leads).all()
        or np.any(leads <= 0)
        or np.any(np.diff(leads) <= 0)
    ):
        raise ValueError(
            f"Forecast leads must be positive and increasing; received {_describe(leads)}. "
            "Supply a nonempty one-dimensional list of increasing positive "
            "multiples of timestep, or a positive number of forecast steps."
        )
    if not isinstance(n_ens_members, (int, np.integer)) or n_ens_members < 1:
        raise ValueError(
            f"n_ens_members must be a positive integer; received {n_ens_members!r}"
        )
    if len(shape) != 2 or any(
        not isinstance(size, (int, np.integer)) or size < 1 for size in shape
    ):
        raise ValueError(
            f"shape must contain two positive integers (ny, nx); received {shape!r}"
        )
    defaults = {
        "institution": kwargs.get("institution", "the pySTEPS community"),
        **metadata,
    }
    defaults["x1"], defaults["x2"] = sorted(
        [float(metadata["x1"]), float(metadata["x2"])]
    )
    defaults["y1"], defaults["y2"] = sorted(
        [float(metadata["y1"]), float(metadata["y2"])]
    )
    base = convert_input_to_xarray_dataset(
        np.broadcast_to(np.array(np.nan), shape), None, defaults
    )
    variable = base.attrs["precip_var"]
    origin = utc_datetime(startdate)
    times = origin + np.rint(leads * float(timestep) * 60e6).astype("timedelta64[us]")
    base = base.assign_coords(time=("time", times, {"standard_name": "time"}))
    dims = ("time", "y", "x")
    sizes = (len(leads), *shape)
    if n_ens_members > 1 or incremental == "member":
        labels = kwargs.get("ensemble_ids", np.arange(1, n_ens_members + 1))
        if len(labels) != n_ens_members:
            raise ValueError(
                f"ensemble_ids must match n_ens_members={n_ens_members}; "
                f"received {len(labels)} labels. Supply one label per member."
            )
        base = base.assign_coords(
            ens_number=(
                "ens_number",
                labels,
                {"standard_name": "realization", "units": "1"},
            )
        )
        dims, sizes = ("ens_number", *dims), (n_ens_members, *sizes)
    attrs = base[variable].attrs.copy()
    attrs["cell_methods"] = metadata.get("cell_methods", attrs["cell_methods"])
    base[variable] = (
        dims,
        np.broadcast_to(
            np.array(
                np.nan, dtype=datatype if np.dtype(datatype).kind == "f" else "float64"
            ),
            sizes,
        ),
        attrs,
    )
    method = temporal_method(base, variable)
    bounds = None
    if method != "point":
        minutes = metadata.get("accutime")
        if not _real_scalar(minutes) or not np.isfinite(minutes) or minutes <= 0:
            raise ValueError(
                "Interval forecast export requires metadata accutime; timestep is "
                f"not duration. Received accutime={minutes!r}. Supply the actual "
                "positive accumulation/averaging duration in minutes."
            )
        bounds = np.column_stack(
            [times - np.timedelta64(int(round(minutes * 60e6)), "us"), times]
        )
    base = set_time_metadata(base, variable, bounds, method)
    base = base.assign_coords(
        forecast_reference_time=xr.DataArray(
            origin, attrs={"standard_name": "forecast_reference_time"}
        )
    )
    base.attrs.update(
        references=kwargs.get("references", ""), comment=kwargs.get("comment", "")
    )
    prepared = _prepare(
        base, include_latlon=kwargs.get("include_latlon", False), validate_values=False
    )
    dtype = np.dtype(datatype)
    fill = (
        fill_value
        if fill_value is not None
        else _netcdf4().default_fillvals[dtype.str[1:]]
    )
    encoding = {variable: {"dtype": dtype, "_FillValue": fill, "complevel": complevel}}
    for key in ("zlib", "shuffle", "fletcher32", "chunksizes"):
        if key in kwargs:
            encoding[variable][key] = kwargs[key]
    if scale_factor is not None:
        encoding[variable]["scale_factor"] = scale_factor
    if offset is not None:
        encoding[variable]["add_offset"] = offset
    options = _encoding(prepared, encoding, legacy=True)
    writer = _CFWriter(
        prepared,
        Path(outpath) / (outfnprefix + ".nc"),
        options,
        overwrite=kwargs.get("overwrite", False),
        incremental=incremental,
    )
    return {
        "method": "netcdf",
        "_cf_writer": writer,
        "ncfile": writer.ncfile,
        "var_F": writer.ncfile[variable],
        "var_time": writer.ncfile["time"],
        "var_name": variable,
        "startdate": startdate,
        "timestep": timestep,
        "metadata": metadata,
        "incremental": incremental,
        "num_timesteps": len(leads),
        "timesteps": n_timesteps,
        "num_ens_members": n_ens_members,
        "shape": tuple(shape),
    }
